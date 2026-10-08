"""Streamlit web interface for YOLO image detection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st


MODEL_FILE = "yolov8n.onnx"
DEFAULT_CLASSES = ["person", "bus", "car", "truck", "bicycle", "motorbike"]
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
GRID_SIZES = (80, 40, 20)
ANCHORS = ((10, 13), (16, 30), (19, 70))
CLASS_COUNT = len(CLASS_NAMES)


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(MODEL_FILE)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}. Commit yolov8n.onnx through Git LFS.")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_left, first_top, first_right, first_bottom = first
    second_left, second_top, second_right, second_bottom = second
    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)
    intersection = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
    first_area = max(0, first_right - first_left) * max(0, first_bottom - first_top)
    second_area = max(0, second_right - second_left) * max(0, second_bottom - second_top)
    return intersection / (first_area + second_area - intersection) if first_area + second_area else 0.0


def non_max_suppression(boxes: list[tuple[int, int, int, int, float, str]], threshold: float) -> list[tuple[int, int, int, int, float, str]]:
    boxes = sorted(boxes, key=lambda box: box[4], reverse=True)
    kept = []
    for box in boxes:
        if all(box_iou(box[:4], candidate[:4]) <= threshold for candidate in kept):
            kept.append(box)
    return kept


def detect_image(image: np.ndarray, model, confidence_threshold: float):
    height, width = image.shape[:2]
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (640, 640))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))[None, ...]

    output = model.run(None, {model.get_inputs()[0].name: input_image})[0]
    predictions = output[0].T
    boxes = []
    cell_offset = 0
    for grid_size, anchor in zip(GRID_SIZES, ANCHORS):
        cell_count = grid_size * grid_size
        for cell_index in range(cell_count):
            anchor_index = cell_offset + cell_index
            raw_box = predictions[anchor_index, :4]
            class_scores = predictions[anchor_index, 4:]
            class_index = int(np.argmax(class_scores))
            confidence = float(sigmoid(class_scores[class_index]))
            if confidence < confidence_threshold:
                continue

            center_x, center_y, box_width, box_height = map(float, raw_box)
            left = max(0, int(center_x - box_width / 2))
            top = max(0, int(center_y - box_height / 2))
            right = min(width, int(center_x + box_width / 2))
            bottom = min(height, int(center_y + box_height / 2))
            boxes.append((left, top, right, bottom, confidence, CLASS_NAMES[class_index]))
        cell_offset += cell_count

    return non_max_suppression(boxes, 0.45)


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
    confidence = st.sidebar.slider("Confidence", min_value=0.20, max_value=0.95, value=0.55)

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
