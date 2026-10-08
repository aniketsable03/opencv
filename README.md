# YOLOv8 Image Detector

This Streamlit application detects people, vehicles, and other objects with the YOLOv8 small model. Upload a JPG, PNG, or WebP image to view annotated results and a detection table.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model

The application uses the Git LFS-tracked `yolov8n.onnx` model. It is approximately 12.4 MB and is installed with the repository.

To export the model locally:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=17, dynamic=True)"
```

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

- The application uses YOLOv8 through ONNX Runtime.
- The model runs at 640 × 640 input resolution.
- The default class filter includes person, bus, car, truck, bicycle, and motorbike.
- The legacy `yolo_detect.py` remains available as a standalone OpenCV CLI detector.
- Streamlit Cloud uses Python 3.14, so the dependency file does not install PyTorch.
