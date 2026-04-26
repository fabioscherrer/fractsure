"""Streamlit frontend that delegates inference to the API service."""

from __future__ import annotations

import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(page_title="Fracture Detection", page_icon="X", layout="centered")
st.title("X-ray Fracture Detection")
st.caption("Upload an X-ray image and run detection through the FastAPI backend.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run Detection", type="primary"):
        with st.spinner("Requesting prediction from backend..."):
            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type or "application/octet-stream",
                )
            }
            try:
                response = requests.post(PREDICT_URL, files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
            else:
                output = image.copy()
                draw = ImageDraw.Draw(output)

                boxes = result.get("boxes", [])
                for box in boxes:
                    x1 = int(box.get("x1", 0))
                    y1 = int(box.get("y1", 0))
                    x2 = int(box.get("x2", 0))
                    y2 = int(box.get("y2", 0))
                    label = box.get("label", "fracture")
                    score = float(box.get("score", 0.0))

                    draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
                    draw.text((x1, max(0, y1 - 18)), f"{label} {score:.2f}", fill="red")

                st.image(output, caption="Prediction result", use_container_width=True)
                st.json(result)
