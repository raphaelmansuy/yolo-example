from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model

# Perform tracking with the model
results = model.track(source="https://www.youtube.com/watch?v=hBwni2ghsMo", show=True)  # Tracking with default tracker