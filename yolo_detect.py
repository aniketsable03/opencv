#!/usr/bin/env python3
"""Run a YOLO detector with OpenCV on a video file or webcam."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people, vehicles, and other objects with YOLO and OpenCV."
    )
    parser.add_argument("--config", required=True, help="YOLO Darknet configuration file")
    parser.add_argument("--weights", required=True, help="YOLO trained weights file")
    parser.add_argument(
        "--source",
        default="0",
        help="Video file path, or webcam index such as 0 (default: 0)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional labels file, one class name per line",
    )
    parser.add_argument(
        "--classes",
        default="person,bus,car,truck,bicycle,motorbike",
        help="Comma-separated classes to display (default: person,bus,car,truck,bicycle,motorbike)",
    )
    parser.add_argument("--confidence", type=float, default=0.50, help="Minimum confidence")
    parser.add_argument("--nms", type=float, default=0.45, help="Non-maximum suppression threshold")
    parser.add_argument("--output", default=None, help="Optional output video path")
    parser.add_argument("--no-display", action="store_true", help="Do not open a window")
    return parser.parse_args()


def load_labels(path: str | None) -> list[str]:
    if path:
        labels = Path(path).read_text(encoding="utf-8").splitlines()
        labels = [label.strip() for label in labels if label.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


DEFAULT_ANCHORS = [
    [10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    [59, 119], [116, 90], [156, 198], [373, 322], [215, 186],
]


def read_anchors(config_path: str) -> list[list[float]]:
    """Read YOLO anchors from a Darknet configuration file."""
    config = Path(config_path).read_text(encoding="utf-8")
    match = re.search(r"^anchors\s*=\s*(.+)$", config, re.MULTILINE)
    if not match:
        return DEFAULT_ANCHORS

    values = [
        float(value.strip())
        for value in match.group(1).replace("\n", ",").split(",")
        if value.strip()
    ]
    if len(values) % 2:
        raise ValueError("The YOLO anchors must contain width and height pairs")
    return [[values[index], values[index + 1]] for index in range(0, len(values), 2)]


def parse_yolo_output(
    outputs: Iterable[np.ndarray],
    labels: list[str],
    anchors: list[list[float]],
    confidence_threshold: float,
    input_size: tuple[int, int] = (416, 416),
) -> list[tuple[int, int, int, int, float, str]]:
    boxes: list[tuple[int, int, int, int, float, str]] = []
    class_count = len(labels)
    input_width, input_height = input_size
    output_channels = 5 + class_count

    for output_index, output in enumerate(outputs):
        data = np.asarray(output)
        if data.ndim == 5:
            data = data[0]

        if data.ndim == 4:
            detection_count, grid_height, grid_width, channels = data.shape
            if channels < output_channels:
                continue
            rows = [data[detection, y, x] for detection in range(detection_count) for y in range(grid_height) for x in range(grid_width)]
            grid_dimensions = (grid_height, grid_width)
        elif data.ndim == 3:
            data = data[0] if data.shape[0] == 1 else data
            if data.ndim != 2 or data.shape[1] < output_channels:
                continue
            rows = data
            anchor_count = len(anchors)
            grid_dimensions = (int(round(np.sqrt(len(rows) / anchor_count))),) * 2
        elif data.ndim == 2:
            rows = data
            anchor_count = len(anchors)
            grid_dimensions = (int(round(np.sqrt(len(rows) / anchor_count))),) * 2
        else:
            continue

        anchor_count = min(len(anchors), max(1, len(rows) // max(1, grid_dimensions[0] * grid_dimensions[1])))
        for row_index, values in enumerate(rows):
            anchor = anchors[min(output_index * anchor_count + row_index // max(1, grid_dimensions[0] * grid_dimensions[1]), len(anchors) - 1)]
            objectness = float(values[4])
            class_scores = values[5:]
            class_index = int(np.argmax(class_scores))
            confidence = float(objectness * class_scores[class_index])
            if confidence < confidence_threshold:
                continue

            grid_index = row_index % (grid_dimensions[0] * grid_dimensions[1])
            y, x = divmod(grid_index, grid_dimensions[1])
            center_x = (x + 0.5) * input_width / grid_dimensions[1]
            center_y = (y + 0.5) * input_height / grid_dimensions[0]
            box_width = anchor[0] * (2.0 * input_width / 416) * np.exp(float(values[0]))
            box_height = anchor[1] * (2.0 * input_height / 416) * np.exp(float(values[1]))
            left = max(0, int(center_x - box_width / 2))
            top = max(0, int(center_y - box_height / 2))
            right = min(input_width, int(center_x + box_width / 2))
            bottom = min(input_height, int(center_y + box_height / 2))
            boxes.append((left, top, right, bottom, confidence, labels[class_index]))

    return boxes


def non_max_suppression(
    boxes: list[tuple[int, int, int, int, float, str]],
    threshold: float,
) -> list[tuple[int, int, int, int, float, str]]:
    selected: list[tuple[int, int, int, int, float, str]] = []
    ordered = sorted(boxes, key=lambda box: box[4], reverse=True)

    for box in ordered:
        left, top, right, bottom, _, _ = box
        overlaps = False
        for selected_box in selected:
            selected_left, selected_top, selected_right, selected_bottom, _, _ = selected_box
            intersection_width = max(0, min(right, selected_right) - max(left, selected_left))
            intersection_height = max(0, min(bottom, selected_bottom) - max(top, selected_top))
            intersection = intersection_width * intersection_height
            area = (right - left) * (bottom - top)
            selected_area = (selected_right - selected_left) * (selected_bottom - selected_top)
            if intersection / (area + selected_area - intersection) >= threshold:
                overlaps = True
                break
        if not overlaps:
            selected.append(box)
    return selected


def draw_box(frame: np.ndarray, box: tuple[int, int, int, int, float, str]) -> None:
    left, top, right, bottom, confidence, label = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(frame, text, (left, max(0, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    selected_classes = {item.strip().lower() for item in args.classes.split(",") if item.strip()}
    anchors = read_anchors(args.config)

    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    layer_names = net.getLayerNames()
    output_layers = [name for name in layer_names if name.startswith("detection_")]
    if not output_layers:
        raise RuntimeError("No YOLO detection layers were found in the configuration file")

    source = args.source
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input: {source}")

    writer = None
    if args.output:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    while True:
        success, frame = capture.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(416, 416),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        outputs = net.forward(output_layers)
        boxes = parse_yolo_output(outputs, labels, anchors, args.confidence)
        boxes = [box for box in boxes if box[5].lower() in selected_classes]
        boxes = non_max_suppression(boxes, args.nms)

        for box in boxes:
            draw_box(frame, box)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
