# YOLOv8 Image Detector

This Streamlit application detects people, vehicles, and other objects with the YOLOv8 small model. Upload a JPG, PNG, or WebP image to view annotated results and a detection table.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download the model

The application uses `yolov8n.pt` in the project directory. It is downloaded automatically when you run:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The model is approximately 6.3 MB and is excluded from Git by `.gitignore`.

## Run the application

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

## Features

- Upload JPG, PNG, or WebP images.
- Select the classes to detect.
- Adjust the confidence threshold from 0.20 to 0.95.
- View the original image with green detection boxes.
- Inspect class, confidence, and bounding-box coordinates in a table.
- Run on CPU without a CUDA installation.

## Model notes

- The application uses Ultralytics YOLOv8.
- The model runs at 640 × 640 input resolution.
- The default class filter includes person, bus, car, truck, bicycle, and motorbike.
- The legacy `yolo_detect.py` remains available as a standalone OpenCV CLI detector.
