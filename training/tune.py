import os
from pathlib import Path
import csv
import requests
import yaml
import mlflow
from ultralytics import YOLO
from ray import tune


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _best_params_from_ray(results, search_keys: set) -> dict | None:
    try:
        best = results.get_best_result(metric="metrics/mAP50(B)", mode="max")
        cfg = best.config or {}
        params = {k: cfg[k] for k in search_keys if k in cfg}
        if params:
            return params
    except Exception as exc:
        print(f"[tune] Ray get_best_result failed ({exc}), falling back to YOLO results.csv")
    return None


def _best_params_from_csv(search_keys: set) -> dict | None:
    detect_root = Path("runs") / "detect"
    if not detect_root.exists():
        return None

    best_map50 = -1.0
    best_args_yaml: Path | None = None

    for trial_dir in sorted(detect_root.glob("tune*")):
        results_csv = trial_dir / "results.csv"
        args_yaml = trial_dir / "args.yaml"
        if not results_csv.exists() or not args_yaml.exists():
            continue
        try:
            with results_csv.open() as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            map50 = float(rows[-1].get("metrics/mAP50(B)", -1))
            if map50 > best_map50:
                best_map50 = map50
                best_args_yaml = args_yaml
        except Exception:
            continue

    if best_args_yaml is None:
        return None

    with best_args_yaml.open() as f:
        args = yaml.safe_load(f)

    params = {k: args[k] for k in search_keys if k in args}
    print(f"[tune] Fallback: best trial {best_args_yaml.parent.name} mAP50={best_map50:.4f}, params={params}")
    return params if params else None


def run_tuning():
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_config(config_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = "Fracture_Detection_Tuning"

    os.environ["ULTRALYTICS_MLFLOW"] = "True"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Diagnose
    try:
        r = requests.get(tracking_uri, timeout=3)
        print(f"[mlflow] Server erreichbar: {r.status_code}")
    except Exception as e:
        print(f"[mlflow] SERVER NICHT ERREICHBAR: {e}")
        raise SystemExit(1)

    model = YOLO(cfg["model"])

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(cfg["data_config"])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()

    print(f"Starte Ray Tune Suche. Tracking an: {tracking_uri}")
    print(f"Datensatz: {data_path}")

    search_space = {
        "lr0":     tune.loguniform(1e-4, 1e-2),
        "degrees": tune.uniform(0.0, 30.0),
        "scale":   tune.uniform(0.4, 0.6),
        "mosaic":  tune.uniform(0.0, 1.0),
        "batch":   tune.choice([4, 8]),        # CPU: [4, 8] — GPU: [16, 32, 64]
    }

    # FIX: Parent-Run ID als Env-Var setzen damit Ray-Worker-Runs als Child erscheinen
    with mlflow.start_run(run_name="ray_tune_search") as parent_run:
        mlflow.log_params({
            "model":            cfg["model"],
            "epochs_per_trial": 1,             # 15 im echten Lauf
            "iterations":       1,             # 20 im echten Lauf
            "optimizer":        "AdamW",
            "data_config":      str(data_path),
        })
        # Ray-Worker erbt diese ID → Ultralytics-Runs werden als Child geloggt
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id

        results = model.tune(
            data=str(data_path),
            epochs=1,        # 15
            iterations=1,    # 20
            use_ray=True,
            gpu_per_trial=0, # bei GPU → 1
            space=search_space,
            optimizer="AdamW",
            workers=4,
        )

    search_keys = set(search_space.keys())

    best_params = _best_params_from_ray(results, search_keys)
    if best_params is None:
        best_params = _best_params_from_csv(search_keys)
    if best_params is None:
        print("[tune] WARNING: could not determine best params — config.yaml not updated.")
        return

    # FIX: 'data' und andere Nicht-Hyperparameter rausfiltern
    for key in ("data", "batch"):
        best_params.pop(key, None)

    if "lr0" in best_params:
        best_params["learning_rate"] = best_params["lr0"]

    with open(config_path, "r", encoding="utf-8") as f:
        current_config = yaml.safe_load(f)

    current_config.update(best_params)
    current_config["run_name"] = "final_model_after_tuning"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)

    print(f"[tune] config.yaml updated: {best_params}")

    with mlflow.start_run(run_name="best_params_summary", nested=False):
        mlflow.log_params(best_params)
        print("[tune] Beste Parameter in MLflow geloggt.")


if __name__ == "__main__":
    run_tuning()