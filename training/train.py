"""YOLO training entrypoint with baseline MLflow integration."""

from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO, settings as yolo_settings


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def disable_ultralytics_mlflow() -> None:
    """Avoid conflicts between Ultralytics automatic MLflow and our manual logging."""
    yolo_settings.update({"mlflow": False})


def cleanup_label_cache(data_config_path: str) -> None:
    """Remove generated labels.cache files from the DVC-tracked dataset tree."""
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = Path(data_config_path)
    if not data_yaml.is_absolute():
        data_yaml = (project_root / data_yaml).resolve()

    if not data_yaml.exists():
        return

    with data_yaml.open("r", encoding="utf-8") as file:
        data_cfg = yaml.safe_load(file)

    for split_key in ("train", "val", "test"):
        split_path = data_cfg.get(split_key)
        if not split_path:
            continue

        images_dir = Path(split_path)
        if not images_dir.is_absolute():
            images_dir = (data_yaml.parent / images_dir).resolve()

        cache_file = images_dir.parent / "labels" / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()


def train() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_config(config_path)

    disable_ultralytics_mlflow()

    mlflow.set_experiment(cfg.get("experiment_name", "fracture-yolo"))

    with mlflow.start_run(run_name=cfg.get("run_name", "baseline")):
        mlflow.log_params(
            {
                "model": cfg["model"],
                "epochs": cfg["epochs"],
                "imgsz": cfg["imgsz"],
                "batch": cfg["batch"],
                "device": cfg["device"],
                "learning_rate": cfg["learning_rate"],
                "weight_decay": cfg["weight_decay"],
                "data_config": cfg["data_config"],
            }
        )

        model = YOLO(cfg["model"])

        try:
            results = model.train(
                data=cfg["data_config"],
                epochs=cfg["epochs"],
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                device=cfg["device"],
                lr0=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )

            # YOLO returns metrics in `results_dict`; keys vary by task/model.
            for key, value in getattr(results, "results_dict", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))

            mlflow.log_param("train_status", "completed")
            print("Training complete and metrics logged to MLflow.")
        except KeyboardInterrupt:
            mlflow.log_param("train_status", "interrupted")
            mlflow.log_text("Training interrupted by user.", "train_error.txt")
            raise
        except Exception as exc:
            mlflow.log_param("train_status", "failed")
            mlflow.log_text(str(exc), "train_error.txt")
            raise
        finally:
            cleanup_label_cache(cfg["data_config"])


if __name__ == "__main__":
    train()
