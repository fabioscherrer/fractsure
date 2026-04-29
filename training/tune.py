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

    # 5. Beste Parameter direkt in die config.yaml schreiben
    print("Beste Parameter gefunden, aktualisiere config.yaml...")
    
    # Die aktuelle Config laden
    with open(config_path, "r") as f:
        current_config = yaml.safe_load(f)

    # Die Parameter in der Config mit den besten Ergebnissen aktualisieren
    # .update() ersetzt vorhandene Werte durch die neuen aus results.best_params
    current_config.update(results.best_params)

    # Die aktualisierte Config zurückschreiben
    with open(config_path, "w") as f:
        yaml.dump(current_config, f, default_flow_style=False)
        
    print(f"config.yaml wurde erfolgreich mit {len(results.best_params)} Parametern aktualisiert.")   


if __name__ == "__main__":
    run_tuning()
    