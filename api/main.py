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

# Class names from data/raw/hbfmid/data.yaml
CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Linear", "Oblique Displaced",
    "Oblique", "Segmental", "Spiral", "Transverse Displaced", "Transverse",
]

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

model_session: ort.InferenceSession | None = None
input_name: str | None = None


def resolve_model_path() -> Path | None:
    if MODEL_PATH_ENV:
        candidate = Path(MODEL_PATH_ENV)
        return candidate if candidate.exists() else None

    model_files = sorted(MODEL_DIR.glob("*.onnx"))
    return model_files[0] if model_files else None


def letterbox(image: np.ndarray, new_size: int = INPUT_SIZE) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image to new_size x new_size keeping aspect ratio, pad with gray (114).
    Returns (padded_image, scale, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = np.array(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    )

    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2

    padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def preprocess(payload: bytes) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    """Bytes -> model input tensor. Returns (tensor, scale, pad, original (w,h))."""
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    orig_w, orig_h = image.size
    img_array = np.array(image)

    padded, scale, pad = letterbox(img_array, INPUT_SIZE)

    # HWC uint8 -> CHW float32 normalized to [0,1] with batch dim
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
    return tensor, scale, pad, (orig_w, orig_h)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Max Suppression. boxes: [N,4] xyxy, scores: [N]. Returns kept indices."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_threshold]
    return keep


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Decode YOLO output (1, 4+nc, N) -> list of detection dicts in original image coords."""
    # (1, 14, 8400) -> (8400, 14)
    preds = output[0].T

    # First 4 cols = box (cx, cy, w, h in 640 space), rest = per-class scores
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= CONF_THRESHOLD
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # cxcywh -> xyxy
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Per-class NMS
    keep_all: list[int] = []
    for cls in np.unique(class_ids):
        idxs = np.where(class_ids == cls)[0]
        kept_local = nms(boxes_xyxy[idxs], confidences[idxs], IOU_THRESHOLD)
        keep_all.extend(idxs[kept_local].tolist())

    if not keep_all:
        return []

    boxes_xyxy = boxes_xyxy[keep_all]
    class_ids = class_ids[keep_all]
    confidences = confidences[keep_all]

    # Undo letterbox: subtract pad, divide by scale
    pad_x, pad_y = pad
    boxes_xyxy[:, [0, 2]] -= pad_x
    boxes_xyxy[:, [1, 3]] -= pad_y
    boxes_xyxy /= scale

    orig_w, orig_h = orig_size
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)

    detections = []
    for box, cls, score in zip(boxes_xyxy, class_ids, confidences):
        detections.append({
            "x1": int(box[0]), "y1": int(box[1]),
            "x2": int(box[2]), "y2": int(box[3]),
            "label": CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls)),
            "score": float(score),
        })
    return detections


@app.on_event("startup")
def load_model() -> None:
    global model_session, input_name

    model_path = resolve_model_path()
    if model_path is None:
        print("No ONNX model found in api/model. /predict will return an error.")
        return

    model_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = model_session.get_inputs()[0].name
    print(f"Loaded ONNX model: {model_path} (input: {input_name})")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(model_session is not None)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if model_session is None or input_name is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are accepted.")

    payload = await file.read()

    try:
        tensor, scale, pad, orig_size = preprocess(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.") from exc

    outputs = model_session.run(None, {input_name: tensor})
    detections = postprocess(outputs[0], scale, pad, orig_size)

    return {
        "model_loaded": True,
        "image_size": {"width": orig_size[0], "height": orig_size[1]},
        "boxes": detections,
    }
