"""Streamlit web interface for YOLO image detection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


MODEL_FILE = "yolov8n.pt"
DEFAULT_CLASSES = ["person", "bus", "car", "truck", "bicycle", "motorbike"]


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(MODEL_FILE)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}. Download yolov8n.pt.")
    return YOLO(str(model_path))


def detect_image(image: np.ndarray, model, confidence_threshold: float):
    results = model.predict(
        image,
        conf=confidence_threshold,
        imgsz=640,
        verbose=False,
        save=False,
        stream=False,
    )
    boxes = []
    for result in results:
        for box, confidence, class_name in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            left, top, right, bottom = map(int, box)
            boxes.append((left, top, right, bottom, float(confidence), str(result.names[int(class_name)])))
    return boxes


def draw_detections(image: np.ndarray, boxes: list[tuple[int, int, int, int, float, str]]) -> np.ndarray:
    result = image.copy()
    for left, top, right, bottom, confidence, label in boxes:
        cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            result,
            f"{label} {confidence:.2f}",
            (left, max(0, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return result


def main() -> None:
    st.set_page_config(page_title="YOLO Image Detector", page_icon="📷", layout="wide")
    st.title("📷 YOLO Image Detector")
    st.caption("Upload a photo to detect people, buses, vehicles, and other objects.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Choose a JPG, PNG, or WebP image.",
    )

    try:
        model = load_model()
    except Exception as error:
        st.error(f"Unable to load the YOLO model: {error}")
        st.info("Run `python -m pip install -r requirements.txt` and ensure yolov8n.pt exists.")
        return

    if uploaded_file is None:
        st.info("Upload a photo to begin detection.")
        return

    image_bytes = uploaded_file.getvalue()
    image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        st.error("The uploaded file could not be read as an image.")
        return

    selected_classes = st.sidebar.multiselect(
        "Detect these classes",
        options=DEFAULT_CLASSES,
        default=DEFAULT_CLASSES,
    )
    confidence = st.sidebar.slider("Confidence", min_value=0.20, max_value=0.95, value=0.50)

    with st.spinner("Detecting objects..."):
        boxes = detect_image(image_array, model, confidence)
        boxes = [box for box in boxes if box[5].lower() in {item.lower() for item in selected_classes}]

    result = draw_detections(image_array, boxes)
    st.subheader("Detection Result")
    st.image(result, channels="BGR", caption=f"Detected {len(boxes)} object(s)")

    if boxes:
        rows = [
            {
                "Class": label,
                "Confidence": round(confidence, 3),
                "Left": left,
                "Top": top,
                "Right": right,
                "Bottom": bottom,
            }
            for left, top, right, bottom, confidence, label in boxes
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.success("No objects matched the selected classes or confidence threshold.")


if __name__ == "__main__":
    main()
