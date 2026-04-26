"""FastAPI inference service for fracture detection."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Fracture Detection API", version="0.1.0")

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
model_session: ort.InferenceSession | None = None

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.01"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "50"))


def resolve_model_path() -> Path | None:
    if MODEL_PATH_ENV:
        candidate = Path(MODEL_PATH_ENV)
        return candidate if candidate.exists() else None

    model_files = sorted(MODEL_DIR.glob("*.onnx"))
    return model_files[0] if model_files else None


def load_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.") from exc


def resolve_input_size(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    default_w, default_h = 640, 640

    if len(shape) < 4:
        return default_w, default_h

    h = shape[2] if isinstance(shape[2], int) else default_h
    w = shape[3] if isinstance(shape[3], int) else default_w
    return int(w), int(h)


def preprocess_image(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize(input_size, resample)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1c = int(np.clip(round(x1), 0, width - 1))
    y1c = int(np.clip(round(y1), 0, height - 1))
    x2c = int(np.clip(round(x2), 0, width - 1))
    y2c = int(np.clip(round(y2), 0, height - 1))

    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


def make_detection(x1: int, y1: int, x2: int, y2: int, class_id: int, confidence: float) -> dict[str, Any]:
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": f"class_{class_id}",
        "score": round(confidence, 4),
    }


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, max_detections: int) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        overlaps = iou(boxes[current], boxes[remaining])
        order = remaining[overlaps < threshold]

    return keep


def normalize_output(raw_outputs: list[np.ndarray]) -> np.ndarray | None:
    for output in raw_outputs:
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1], -1).T

        if arr.ndim != 2 or arr.size == 0:
            continue

        # Handle common YOLO export layout: (84, N) -> (N, 84)
        if arr.shape[0] <= 128 and arr.shape[1] > 128:
            arr = arr.T

        if arr.shape[1] < 5 and arr.shape[0] >= 5:
            arr = arr.T

        if arr.shape[1] >= 5:
            return arr.astype(np.float32, copy=False)

    return None


def decode_xywh(rows: np.ndarray, input_size: tuple[int, int], original_size: tuple[int, int]) -> list[dict[str, Any]]:
    input_w, input_h = input_size
    original_w, original_h = original_size
    sx = original_w / input_w
    sy = original_h / input_h

    detections_with_obj: list[dict[str, Any]] = []
    detections_without_obj: list[dict[str, Any]] = []
    for row in rows:
        if row.shape[0] < 5:
            continue

        cx, cy, w, h = map(float, row[:4])
        tail = row[4:]
        if tail.size == 0:
            continue

        x1 = (cx - w / 2.0) * sx
        y1 = (cy - h / 2.0) * sy
        x2 = (cx + w / 2.0) * sx
        y2 = (cy + h / 2.0) * sy
        clipped = clip_box(x1, y1, x2, y2, original_w, original_h)
        if clipped is None:
            continue

        x1c, y1c, x2c, y2c = clipped
        class_id = int(np.argmax(tail))
        confidence = float(tail[class_id])
        if confidence >= CONF_THRESHOLD:
            detections_without_obj.append(make_detection(x1c, y1c, x2c, y2c, class_id, confidence))

        if tail.size >= 2:
            objectness = float(tail[0])
            class_scores = tail[1:]
            class_id_obj = int(np.argmax(class_scores))
            confidence_obj = objectness * float(class_scores[class_id_obj])
            if confidence_obj >= CONF_THRESHOLD:
                detections_with_obj.append(
                    make_detection(x1c, y1c, x2c, y2c, class_id_obj, confidence_obj)
                )

    # Different exports include objectness differently. Use the richer result set.
    return detections_with_obj if len(detections_with_obj) >= len(detections_without_obj) else detections_without_obj


def decode_xyxy(rows: np.ndarray, input_size: tuple[int, int], original_size: tuple[int, int]) -> list[dict[str, Any]]:
    input_w, input_h = input_size
    original_w, original_h = original_size

    detections: list[dict[str, Any]] = []
    for row in rows:
        if row.shape[0] < 6:
            continue

        x1, y1, x2, y2, confidence, class_id = map(float, row[:6])
        if confidence < CONF_THRESHOLD:
            continue

        # Some exports return normalized coordinates, others return model-space coordinates.
        if max(x1, y1, x2, y2) <= 1.5:
            x1 *= original_w
            x2 *= original_w
            y1 *= original_h
            y2 *= original_h
        else:
            x1 *= original_w / input_w
            x2 *= original_w / input_w
            y1 *= original_h / input_h
            y2 *= original_h / input_h

        clipped = clip_box(x1, y1, x2, y2, original_w, original_h)
        if clipped is None:
            continue

        x1c, y1c, x2c, y2c = clipped
        class_index = int(round(class_id))
        detections.append(make_detection(x1c, y1c, x2c, y2c, class_index, confidence))

    return detections


def decode_detections(
    raw_outputs: list[np.ndarray],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = normalize_output(raw_outputs)
    if rows is None:
        return []

    if rows.shape[1] == 6:
        detections = decode_xyxy(rows, input_size, original_size)
    else:
        detections = decode_xywh(rows, input_size, original_size)

    if not detections:
        return []

    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
    scores = np.array([float(d["score"]) for d in detections], dtype=np.float32)
    keep = nms(boxes, scores, threshold=IOU_THRESHOLD, max_detections=MAX_DETECTIONS)
    return [detections[i] for i in keep]


def placeholder_box(width: int, height: int) -> dict[str, Any]:
    return {
        "x1": int(width * 0.2),
        "y1": int(height * 0.2),
        "x2": int(width * 0.7),
        "y2": int(height * 0.7),
        "label": "possible_fracture",
        "score": 0.76,
    }


@app.on_event("startup")
def load_model() -> None:
    global model_session

    model_path = resolve_model_path()
    if model_path is None:
        print("No ONNX model found in api/model. Using placeholder prediction output.")
        return

    try:
        model_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        print(f"Loaded ONNX model: {model_path}")
    except Exception as exc:
        model_session = None
        print(f"Failed to load ONNX model at {model_path}: {exc}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are accepted.")

    payload = await file.read()
    image = load_image(payload)
    width, height = image.size

    if model_session is None:
        return {
            "model_loaded": False,
            "boxes": [placeholder_box(width, height)],
        }

    try:
        input_name = model_session.get_inputs()[0].name
        input_size = resolve_input_size(model_session)
        input_tensor = preprocess_image(image, input_size)
        raw_outputs = model_session.run(None, {input_name: input_tensor})
        boxes = decode_detections(raw_outputs, input_size=input_size, original_size=(width, height))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return {
        "model_loaded": True,
        "boxes": boxes,
    }
