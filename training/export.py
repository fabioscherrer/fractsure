"""Export best MLflow-tracked YOLO model to ONNX for API inference."""

from pathlib import Path
from mlflow.tracking import MlflowClient
from ultralytics import YOLO


def find_best_run(experiment_name: str, metric_name: str = "metrics/mAP50(B)") -> str:
    # FIX: metric_name muss identisch mit dem Key in train.py sein
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",  # nur abgeschlossene Runs
        order_by=[f"metrics.`{metric_name}` DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No finished runs found in experiment: {experiment_name}")

    best_run = runs[0]
    map50 = best_run.data.metrics.get(metric_name, 0.0)
    print(f"[export] Bester Run: {best_run.info.run_id} — mAP50(B)={map50:.4f}")
    return best_run.info.run_id


def find_latest_local_best(project_root: Path) -> Path | None:
    candidates = sorted(
        project_root.glob("runs/detect/*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"[export] Lokaler Fallback: {candidates[0]}")
    return candidates[0] if candidates else None


def export_best_to_onnx(experiment_name: str = "fracture-yolo") -> Path:
    project_root = Path(__file__).resolve().parents[1]
    client = MlflowClient()
    weights_path: Path | None = None

    try:
        run_id = find_best_run(experiment_name)
        downloaded_weights = client.download_artifacts(run_id, "weights/best.pt")
        weights_path = Path(downloaded_weights)
        print(f"[export] Weights aus MLflow geladen: {weights_path}")
    except Exception as exc:
        print(f"[export] MLflow download fehlgeschlagen ({exc}), nutze lokalen Fallback.")
        weights_path = find_latest_local_best(project_root)

    if weights_path is None:
        raise ValueError(
            "Kein best.pt gefunden — weder in MLflow noch in runs/detect/*/weights/."
        )

    model = YOLO(str(weights_path))
    exported = Path(model.export(format="onnx"))

    target_dir = project_root / "api" / "model"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "fracture_detector.onnx"
    target_path.write_bytes(exported.read_bytes())

    print(f"[export] ONNX Model gespeichert: {target_path}")
    return target_path


if __name__ == "__main__":
    export_best_to_onnx()