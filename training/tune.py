import os # schon im uv?
import pandas as pd
from pathlib import Path
import yaml
import mlflow # schon im uv?
from ultralytics import YOLO
from ray import tune

def check_dataset_integrity(base_path: Path):
    splits = ['train', 'valid', 'test']
    report_data = []
    
    print(f"\n Integritätscheck: {base_path}")
    
    for split in splits:
        img_dir = base_path / split / "images"
        lbl_dir = base_path / split / "labels"
        
        if not img_dir.exists():
            continue
        
        images = {f.stem for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        labels = {f.stem for f in lbl_dir.glob("*") if f.suffix.lower() == '.txt'}
        
        missing = images - labels
        report_data.append({
            "Split": split, "Bilder": len(images), "Labels": len(labels), "Fehlend": len(missing)
        })
        
        if missing:
            print(f"Warnung: {split} hat {len(missing)} Bilder ohne Labels!")

    df = pd.DataFrame(report_data)
    print(df.to_string(index=False))
    print("-" * 30)
    
    # Optional: Abbrechen, wenn zu viele Labels fehlen
    # if df["Fehlend"].sum() > 50:
    #    raise ValueError("Zu viele fehlende Labels! Tuning abgebrochen.")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def run_tuning():
    # 1. Pfade & Config laden
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_config(config_path)

    # 2. Daten prüfen
    data_root = Path("/home/Zenbook-S14-Fedora/code/fractsure/data/raw/hbfmid")
    check_dataset_integrity(data_root)

    # 3. MLflow Setup
    # Hier legst du fest, WO gespeichert wird und WIE das Experiment heißt
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Fracture_Detection_Tuning") # Dein gewählter Name

    # 4. Modell laden
    model = YOLO(cfg["model"])

    print(f"Starte Ray Tune Suche. Tracking an: {tracking_uri}")
    
    # 5. Tuning starten
    # Ultralytics nutzt MLflow automatisch, wenn es initialisiert wurde
    results = model.tune(
        data=cfg["data_config"],
        epochs=1, # 15
        iterations=1, # 20
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

    # 6. Beste Parameter laden und an config.yaml schreiben
    with open(config_path, "r") as f:
        current_config = yaml.safe_load(f)

    best_params = results.best_params
    
    # WICHTIG: Namen angleichen, falls nötig
    # Ray Tune nutzt 'lr0', dein train.py nutzt 'learning_rate'
    if "lr0" in best_params:
        best_params["learning_rate"] = best_params["lr0"]

    # Update und Speichern
    current_config.update(best_params)
    # Wir setzen den Run-Name für das finale Training schon mal fest
    current_config["run_name"] = "final_model_after_tuning"

    with open(config_path, "w") as f:
        yaml.dump(current_config, f, default_flow_style=False)


if __name__ == "__main__":
    run_tuning()
    