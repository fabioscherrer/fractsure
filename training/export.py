"""Export best MLflow-tracked YOLO model to ONNX for API inference."""

from pathlib import Path

from mlflow.tracking import MlflowClient
from ultralytics import YOLO


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
    exported = Path(model.export(format="onnx"))

    target_dir = project_root / "api" / "model"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "fracture_detector.onnx"
    target_path.write_bytes(exported.read_bytes())

    print(f"Exported ONNX model to: {target_path}")
    return target_path


if __name__ == "__main__":
    export_best_to_onnx()
