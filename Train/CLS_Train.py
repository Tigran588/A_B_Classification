from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # Pretrained YOLOv8n classification model

model.train(
    data="path/to/data.yaml",  # Path to dataset config
    epochs=50,                 # Number of epochs
    imgsz=224,                 # Image size (classification typically uses 224x224)
    batch=16,                  # Batch size
    project="runs/train",      # Output directory
    name="exp",                # Experiment name
    device=0                   # GPU device (use -1 for CPU)
)