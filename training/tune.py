import os # schon im uv?
from pathlib import Path
import yaml
import mlflow # schon im uv?
from ultralytics import YOLO
from ray import tune

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def run_tuning():
    # 1. Pfade & Config laden
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_config(config_path)

    # 2. MLflow Setup
    # Hier legst du fest, WO gespeichert wird und WIE das Experiment heißt
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Fracture_Detection_Tuning") # Dein gewählter Name

    # 3. Modell laden
    model = YOLO(cfg["model"])

    print(f"Starte Ray Tune Suche. Tracking an: {tracking_uri}")
    
    # 4. Tuning starten
    # Ultralytics nutzt MLflow automatisch, wenn es initialisiert wurde
    results = model.tune(
        data=cfg["data_config"],
        epochs=15,
        iterations=20,
        use_ray=True,
        gpu_per_trial=0,   # falls kein gpu vorhanden 1 zu 0 ändern
        space={
            "lr0": tune.loguniform(1e-4, 1e-2),
            "degrees": tune.uniform(0.0, 30.0),
            "scale": tune.uniform(0.4, 0.6),
            "mosaic": tune.uniform(0.0, 1.0),
            "batch": tune.choice([4, 8]) # Batch-Optionen für gpu [16, 32, 64]
        }
    )

    # 5. Beste Parameter loggen & speichern
    print("Beste Parameter:")
    print(results.best_params)
    
    # "Parent Run" in MLflow machen, um das Gesamtergebnis zu sichern
    with mlflow.start_run(run_name="tuning_summary"):
        mlflow.log_params(results.best_params)
        mlflow.log_artifact(str(config_path)) # Speichert die genutzte Config

    # Beste Parameter in eine neue YAML schreiben für train.py
    best_params_path = Path(__file__).parent / "best_hyperparameters.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(results.best_params, f)

if __name__ == "__main__":
    run_tuning()
    