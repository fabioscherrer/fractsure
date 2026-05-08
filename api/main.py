"""FastAPI inference service for fracture detection."""

from __future__ import annotations

import ast
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Fracture Detection API", version="0.1.0")

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH_ENV = os.getenv("MODEL_PATH")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.01"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "50"))
DEFAULT_INPUT_SIZE = (640, 640)


@dataclass(frozen=True)
class LetterboxInfo:
    """Information needed to map model-space boxes back to the original image."""

    ratio: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class ModelContext:
    """Loaded ONNX model plus the metadata needed for YOLO post-processing."""

    session: ort.InferenceSession
    input_name: str
    input_size: tuple[int, int]
    class_names: dict[int, str]


model_context: ModelContext | None = None


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


def parse_class_names(session: ort.InferenceSession) -> dict[int, str]:
    """Read Ultralytics class labels from ONNX custom metadata."""

    names_raw = session.get_modelmeta().custom_metadata_map.get("names")
    if not names_raw:
        return {}

    try:
        parsed = ast.literal_eval(names_raw)
    except (SyntaxError, ValueError):
        return {}

    if isinstance(parsed, dict):
        return {int(key): str(value) for key, value in parsed.items()}

    if isinstance(parsed, list):
        return {index: str(value) for index, value in enumerate(parsed)}

    return {}


def resolve_input_size(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    default_w, default_h = DEFAULT_INPUT_SIZE

    if len(shape) < 4:
        return default_w, default_h

    h = shape[2] if isinstance(shape[2], int) else default_h
    w = shape[3] if isinstance(shape[3], int) else default_w
    return int(w), int(h)


def letterbox_image(
    image: Image.Image, input_size: tuple[int, int]
) -> tuple[Image.Image, LetterboxInfo]:
    """Resize without distortion using the same letterbox strategy YOLO expects."""

    input_w, input_h = input_size
    original_w, original_h = image.size
    ratio = min(input_w / original_w, input_h / original_h)
    resized_w = int(round(original_w * ratio))
    resized_h = int(round(original_h * ratio))

    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else 2
    resized = image.resize((resized_w, resized_h), resample)

    canvas = Image.new("RGB", (input_w, input_h), (114, 114, 114))
    pad_x = (input_w - resized_w) / 2
    pad_y = (input_h - resized_h) / 2
    canvas.paste(resized, (int(round(pad_x - 0.1)), int(round(pad_y - 0.1))))

    return canvas, LetterboxInfo(ratio=ratio, pad_x=pad_x, pad_y=pad_y)


def preprocess_image(
    image: Image.Image, input_size: tuple[int, int]
) -> tuple[np.ndarray, LetterboxInfo]:
    letterboxed, letterbox = letterbox_image(image, input_size)
    array = np.asarray(letterboxed, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0), letterbox


def clip_boxes(boxes: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    original_w, original_h = original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
    return boxes


def scale_boxes_to_original(
    boxes: np.ndarray,
    letterbox: LetterboxInfo,
    original_size: tuple[int, int],
) -> np.ndarray:
    """Map xyxy boxes from model input pixels back to original image pixels."""

    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - letterbox.pad_x) / letterbox.ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - letterbox.pad_y) / letterbox.ratio
    return clip_boxes(boxes, original_size)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def make_detection(
    box: np.ndarray, class_id: int, confidence: float, class_names: dict[int, str]
) -> dict[str, Any]:
    x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": class_names.get(class_id, f"class_{class_id}"),
        "score": round(float(confidence), 4),
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

    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        overlaps = iou(boxes[current], boxes[remaining])
        order = remaining[overlaps < threshold]

    return keep


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    threshold: float,
    max_detections: int,
) -> list[int]:
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        class_keep = nms(boxes[class_indices], scores[class_indices], threshold)
        keep.extend(class_indices[index] for index in class_keep)

    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep[:max_detections]


def normalize_output(raw_outputs: list[np.ndarray]) -> np.ndarray | None:
    """Return model predictions as rows, supporting common Ultralytics ONNX layouts."""

    for output in raw_outputs:
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1], -1).T

        if arr.ndim != 2 or arr.size == 0:
            continue

        # Ultralytics detect export without built-in NMS is usually (4 + classes, anchors).
        if arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
            arr = arr.T

        if arr.shape[1] >= 5:
            return arr.astype(np.float32, copy=False)

    return None


def split_scores(rows: np.ndarray, class_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract class ids and confidences from YOLO rows.

    Current Ultralytics YOLO detect exports output ``xywh + class scores``. Older YOLO
    variants can output ``xywh + objectness + class scores``; this supports both when
    the class count can be inferred from ONNX metadata.
    """

    score_columns = rows[:, 4:]

    if class_count > 0 and score_columns.shape[1] == class_count + 1:
        objectness = score_columns[:, 0]
        class_scores = score_columns[:, 1:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        return class_ids.astype(np.int64), scores.astype(np.float32)

    class_scores = score_columns
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    return class_ids.astype(np.int64), scores.astype(np.float32)


def decode_raw_yolo_output(
    rows: np.ndarray,
    letterbox: LetterboxInfo,
    original_size: tuple[int, int],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    class_ids, scores = split_scores(rows, class_count=len(class_names))
    candidates = scores >= CONF_THRESHOLD
    if not np.any(candidates):
        return []

    boxes = xywh_to_xyxy(rows[candidates, :4])
    boxes = scale_boxes_to_original(boxes, letterbox, original_size)
    scores = scores[candidates]
    class_ids = class_ids[candidates]

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    keep = class_aware_nms(
        boxes, scores, class_ids, threshold=IOU_THRESHOLD, max_detections=MAX_DETECTIONS
    )
    return [
        make_detection(
            boxes[index], int(class_ids[index]), float(scores[index]), class_names
        )
        for index in keep
    ]


def decode_nms_output(
    rows: np.ndarray,
    letterbox: LetterboxInfo,
    original_size: tuple[int, int],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Decode an Ultralytics ONNX export that already includes NMS in the graph."""

    rows = rows[rows[:, 4] >= CONF_THRESHOLD]
    if rows.size == 0:
        return []

    boxes = rows[:, :4]
    if float(np.max(boxes)) <= 1.5:
        original_w, original_h = original_size
        boxes = boxes * np.array(
            [original_w, original_h, original_w, original_h], dtype=np.float32
        )
    else:
        boxes = scale_boxes_to_original(boxes, letterbox, original_size)

    scores = rows[:, 4]
    class_ids = np.rint(rows[:, 5]).astype(np.int64)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    detections = [
        make_detection(box, int(class_id), float(score), class_names)
        for box, score, class_id in zip(
            boxes[valid], scores[valid], class_ids[valid], strict=False
        )
    ]
    return detections[:MAX_DETECTIONS]


def decode_detections(
    raw_outputs: list[np.ndarray],
    letterbox: LetterboxInfo,
    original_size: tuple[int, int],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows = normalize_output(raw_outputs)
    if rows is None:
        return []

    if rows.shape[1] == 6:
        return decode_nms_output(rows, letterbox, original_size, class_names)

    return decode_raw_yolo_output(rows, letterbox, original_size, class_names)


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
    global model_context

    model_path = resolve_model_path()
    if model_path is None:
        print("No ONNX model found in api/model. Using placeholder prediction output.")
        return

    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        model_context = ModelContext(
            session=session,
            input_name=session.get_inputs()[0].name,
            input_size=resolve_input_size(session),
            class_names=parse_class_names(session),
        )
        print(f"Loaded ONNX model: {model_path}")
        print(
            f"Input size: {model_context.input_size}; classes: {model_context.class_names or 'unknown'}"
        )
    except Exception as exc:
        model_context = None
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

    if model_context is None:
        return {
            "model_loaded": False,
            "boxes": [placeholder_box(width, height)],
        }

    try:
        input_tensor, letterbox = preprocess_image(image, model_context.input_size)
        outputs = model_context.session.run(
            None, {model_context.input_name: input_tensor}
        )
        raw_outputs = [np.asarray(output) for output in outputs]
        boxes = decode_detections(
            raw_outputs,
            letterbox=letterbox,
            original_size=(width, height),
            class_names=model_context.class_names,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {exc}"
        ) from exc

    return {
        "model_loaded": True,
        "boxes": boxes,
    }
