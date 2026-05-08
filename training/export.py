"""Export best MLflow-tracked YOLO model to ONNX for API inference."""

import os
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient
from ultralytics.models import YOLO


def find_best_run(experiment_name: str, metric_name: str = "metrics.mAP50") -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric_name} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    return runs[0].info.run_id


def find_latest_local_best(project_root: Path) -> Path | None:
    candidates = sorted(
        project_root.glob("runs/detect/*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def read_training_imgsz(project_root: Path, default: int = 640) -> int:
    config_path = project_root / "training" / "config.yaml"
    if not config_path.exists():
        return default

    config = yaml.safe_load(config_path.read_text()) or {}
    return int(config.get("imgsz", default))


def export_best_to_onnx(experiment_name: str = "fracture-yolo") -> Path:
    project_root = Path(__file__).resolve().parents[1]
    client = MlflowClient()
    weights_path: Path | None = None

    try:
        run_id = find_best_run(experiment_name)
        downloaded_weights = client.download_artifacts(run_id, "weights/best.pt")
        weights_path = Path(downloaded_weights)
    except Exception:
        weights_path = find_latest_local_best(project_root)

    if weights_path is None:
        raise ValueError(
            "Could not find best.pt in MLflow artifacts or local runs/detect outputs."
        )

    model = YOLO(str(weights_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=read_training_imgsz(project_root),
            simplify=True,
            nms=env_flag("ONNX_EXPORT_NMS", default=False),
        )
    )

    target_dir = project_root / "api" / "model"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "fracture_detector.onnx"
    target_path.write_bytes(exported.read_bytes())

    print(f"Exported ONNX model to: {target_path}")
    return target_path


if __name__ == "__main__":
    export_best_to_onnx()
