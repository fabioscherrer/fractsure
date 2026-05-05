import os
from pathlib import Path
import csv
import requests
import yaml
import mlflow
from ultralytics import YOLO
from ray import tune
import pandas as pd


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def check_dataset_integrity(base_path: Path) -> None:
    splits = ["train", "valid", "test"]
    report_data = []
    print(f"\n[data] Integritätscheck: {base_path}")

    for split in splits:
        img_dir = base_path / split / "images"
        lbl_dir = base_path / split / "labels"
        if not img_dir.exists():
            continue

        images = {f.stem for f in img_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        labels = {f.stem for f in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        missing = images - labels

        report_data.append({
            "Split": split,
            "Bilder": len(images),
            "Labels": len(labels),
            "Fehlend": len(missing),
        })

        if missing:
            print(f"  ⚠️  {split}: fehlende Labels für: {sorted(missing)[:5]}")

    df = pd.DataFrame(report_data)
    print(df.to_string(index=False))
    print("-" * 40)

    if df["Fehlend"].sum() > 50:
        raise ValueError(f"[data] {df['Fehlend'].sum()} fehlende Labels — Tuning abgebrochen.")

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

    try:
        r = requests.get(tracking_uri, timeout=3)
        print(f"[mlflow] Server erreichbar: {r.status_code}")
    except Exception as e:
        print(f"[mlflow] SERVER NICHT ERREICHBAR: {e}")
        raise SystemExit(1)

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(cfg["data_config"])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()

    # Dataset-Integritätscheck
    check_dataset_integrity(data_path.parent)

    model = YOLO(cfg["model"])

    print(f"Starte Ray Tune Suche. Tracking an: {tracking_uri}")
    print(f"Datensatz: {data_path}")

    search_space = {
        "lr0":          tune.loguniform(1e-4, 1e-2),
        "degrees":      tune.uniform(0.0, 30.0),
        "scale":        tune.uniform(0.4, 0.6),
        "mosaic":       tune.uniform(0.0, 1.0),
        "dropout":      tune.uniform(0.0, 0.3),         # Regularisierung
        "weight_decay": tune.loguniform(1e-5, 1e-3),    # L2-Regularisierung
        "batch":        tune.choice([4, 8]),            # GPU: [16, 32, 64]
    }

    with mlflow.start_run(run_name="ray_tune_search") as parent_run:
        mlflow.log_params({
            "model":            cfg["model"],
            "epochs_per_trial": 1,    # 15 im echten Lauf
            "iterations":       1,    # 20 im echten Lauf
            "optimizer":        "AdamW",
            "data_config":      str(data_path),
        })
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
            # Fix gegen Overfitting — nicht tunen, fix setzen
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            patience=20,     # früh stoppen wenn keine Verbesserung
        )

    search_keys = set(search_space.keys())

    best_params = _best_params_from_ray(results, search_keys)
    if best_params is None:
        best_params = _best_params_from_csv(search_keys)
    if best_params is None:
        print("[tune] WARNING: could not determine best params — config.yaml nicht aktualisiert.")
        return

    # Nicht-Hyperparameter rausfiltern
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