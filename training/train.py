"""YOLO training entrypoint with baseline MLflow integration."""

from pathlib import Path
import os
import mlflow
import yaml
from ultralytics import YOLO, settings as yolo_settings


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def disable_ultralytics_mlflow() -> None:
    """Avoid conflicts between Ultralytics automatic MLflow and our manual logging.

    In train.py loggen wir bewusst manuell (ein einziger fokussierter Run).
    In tune.py ist Ultralytics-Auto-MLflow dagegen aktiv, weil dort jeder Ray-Trial
    automatisch als eigener Run geloggt werden soll.
    """
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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "fracture-yolo"))

    disable_ultralytics_mlflow()

    with mlflow.start_run(run_name=cfg.get("run_name", "baseline")):
        mlflow.log_params({
            "model":          cfg["model"],
            "epochs":         cfg["epochs"],
            "imgsz":          cfg["imgsz"],
            "batch":          cfg["batch"],
            "device":         cfg["device"],
            "learning_rate":  cfg["learning_rate"],
            "weight_decay":   cfg.get("weight_decay", 0.0005),
            "dropout":        cfg.get("dropout", 0.0),
            "degrees":        cfg.get("degrees", 0.0),
            "scale":          cfg.get("scale", 0.5),
            "mosaic":         cfg.get("mosaic", 1.0),
            "data_config":    cfg["data_config"],
        })

        model = YOLO(cfg["model"])
        
        # NEU: Callback für Epoch-für-Epoch Logging
        def log_epoch_metrics(trainer):
            if not trainer.metrics:
                return
            epoch = trainer.epoch
            for key, value in trainer.metrics.items():
                if isinstance(value, (int, float)):
                    # FIX: Klammern ersetzen damit MLflow den Namen akzeptiert
                    clean_key = key.replace("(", "_").replace(")", "")
                    mlflow.log_metric(clean_key, float(value), step=epoch)
            for key, value in trainer.label_loss_items(trainer.tloss, prefix="train").items():
                if isinstance(value, (int, float)):
                    clean_key = key.replace("(", "_").replace(")", "")
                    mlflow.log_metric(clean_key, float(value), step=epoch)

        model.add_callback("on_fit_epoch_end", log_epoch_metrics)

        try:
            results = model.train(
                data=cfg["data_config"],
                epochs=cfg["epochs"],
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                device=cfg["device"],
                lr0=cfg["learning_rate"],
                weight_decay=cfg.get("weight_decay", 0.0005),
                dropout=cfg.get("dropout", 0.0),
                # Augmentierungen aus Tuning
                degrees=cfg.get("degrees", 0.0),
                scale=cfg.get("scale", 0.5),
                mosaic=cfg.get("mosaic", 1.0),
                # Fix gegen Overfitting
                fliplr=0.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                patience=7,      # früh stoppen — wichtiger Anti-Overfitting-Parameter (vorher 15)
                optimizer="AdamW",
                workers=8,       # train.py läuft solo auf der VM (kein Ray) → volle 8 DataLoader-Threads ok
            )

            for key, value in getattr(results, "results_dict", {}).items():
                if isinstance(value, (int, float)):
                    clean_key = key.replace("(", "_").replace(")", "")  # ← Fix hier auch
                    mlflow.log_metric(clean_key, float(value))

            # best.pt unter "weights/" loggen — export.py erwartet genau diesen Pfad.
            weights_dir = Path(results.save_dir) / "weights"
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")
                print(f"[mlflow] best.pt geloggt: {best_pt}")
            else:
                print("[mlflow] WARNUNG: best.pt nicht gefunden, wurde nicht geloggt.")

            # YOLO erzeugt Plots/CSVs/Sample-Bilder in results.save_dir.
            # Wir laden alle Nicht-.pt-Dateien rekursiv unter "yolo_outputs/" hoch,
            # damit Loss-Kurven, Confusion Matrix, PR-/F1-Curve, Sample-Predictions
            # und die Trainings-Augmentierungs-Beispiele zentral im MLflow-UI sichtbar sind.
            # .pt-Files werden ausgeschlossen (best.pt liegt schon unter "weights/", last.pt nicht nötig).
            yolo_run_dir = Path(results.save_dir)
            if yolo_run_dir.exists():
                logged_count = 0
                for file_path in yolo_run_dir.rglob("*"):
                    if not file_path.is_file() or file_path.suffix == ".pt":
                        continue
                    rel_dir = file_path.parent.relative_to(yolo_run_dir)
                    artifact_subdir = "yolo_outputs" if rel_dir == Path(".") else f"yolo_outputs/{rel_dir.as_posix()}"
                    mlflow.log_artifact(str(file_path), artifact_path=artifact_subdir)
                    logged_count += 1
                print(f"[mlflow] YOLO-Outputs hochgeladen: {logged_count} Dateien aus {yolo_run_dir}")

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
