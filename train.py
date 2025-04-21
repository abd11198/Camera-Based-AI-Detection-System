# train.py

from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt depending on your preference

# Train the model
model.train(
    data='data.yaml',       # Path to your data config
    epochs=50,              # You can change this
    imgsz=640,              # Image size
    batch=8,                # Batch size
    name='key-detector',    # Experiment name
)
