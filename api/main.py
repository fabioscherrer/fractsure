"""FastAPI inference service for fracture detection."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Fracture Detection API", version="0.1.0")

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
model_session: ort.InferenceSession | None = None


def resolve_model_path() -> Path | None:
    if MODEL_PATH_ENV:
        candidate = Path(MODEL_PATH_ENV)
        return candidate if candidate.exists() else None

    model_files = sorted(MODEL_DIR.glob("*.onnx"))
    return model_files[0] if model_files else None


def validate_image(payload: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
        return width, height
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.") from exc


@app.on_event("startup")
def load_model() -> None:
    global model_session

    model_path = resolve_model_path()
    if model_path is None:
        print("No ONNX model found in api/model. Using placeholder prediction output.")
        return

    model_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX model: {model_path}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are accepted.")

    payload = await file.read()
    width, height = validate_image(payload)

    # Placeholder response until full pre/post-processing is implemented.
    box = {
        "x1": int(width * 0.2),
        "y1": int(height * 0.2),
        "x2": int(width * 0.7),
        "y2": int(height * 0.7),
        "label": "possible_fracture",
        "score": 0.76,
    }

    return {
        "model_loaded": model_session is not None,
        "boxes": [box],
    }
