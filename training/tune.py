import os
from pathlib import Path

import csv
import yaml
import mlflow
from ultralytics import YOLO
from ray import tune


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _best_params_from_ray(results, search_keys: set) -> dict | None:
    """Try Ray's get_best_result first. Returns None if metrics are unavailable."""
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
    """
    Fallback: read YOLO's per-trial results.csv files from runs/detect/tune*.
    Finds the trial directory with the highest final mAP50 and reads its args.yaml
    to recover the sampled hyperparameters.
    """
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

    # Must be env-vars so Ray workers and Ultralytics' MLflow callback inherit them.
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model = YOLO(cfg["model"])

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(cfg["data_config"])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()

    print(f"Starte Ray Tune Suche. Tracking an: {tracking_uri}")
    print(f"Datensatz: {data_path}")

    search_space = {
        "lr0": tune.loguniform(1e-4, 1e-2),
        "degrees": tune.uniform(0.0, 30.0),
        "scale": tune.uniform(0.4, 0.6),
        "mosaic": tune.uniform(0.0, 1.0),
        "batch": tune.choice([4, 8]),  # GPU: [16, 32, 64]
    }

    results = model.tune(
        data=str(data_path),
        epochs=2,        # SMOKE TEST: set to 15 for GCP
        iterations=2,    # SMOKE TEST: set to 20 for GCP
        use_ray=True,
        gpu_per_trial=0,  # set to 1 on GCP
        space=search_space,
    )

    search_keys = set(search_space.keys())

    # Prefer Ray's result grid; fall back to reading YOLO's results.csv files.
    best_params = _best_params_from_ray(results, search_keys)
    if best_params is None:
        best_params = _best_params_from_csv(search_keys)
    if best_params is None:
        print("[tune] WARNING: could not determine best params — config.yaml not updated.")
        return

    # lr0 → learning_rate alias for train.py
    if "lr0" in best_params:
        best_params["learning_rate"] = best_params["lr0"]

    with open(config_path, "r") as f:
        current_config = yaml.safe_load(f)

    current_config.update(best_params)
    current_config["run_name"] = "final_model_after_tuning"

    with open(config_path, "w") as f:
        yaml.dump(current_config, f, default_flow_style=False)

    print(f"[tune] config.yaml updated: {best_params}")


if __name__ == "__main__":
    run_tuning()
