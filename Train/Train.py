from ultralytics import YOLO
import datetime
import os
import torch


class Training:
    def __init__(self, data_path, yolo_weights='yolov8n.pt', num_epochs=100, img_size=640):
        """
        Initialize training configuration.
        :param data_path: Path to data.yaml file
        :param yolo_weights: Pretrained weights (default: yolov8n.pt)
        :param num_epochs: Number of epochs to train
        :param img_size: Image size for training (default: 640, good for 480x640 dataset)
        """
        self.weights = yolo_weights
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.img_size = img_size

    def train(self):
        # Logging directory
        log_dir = 'logs/tensorboard_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        print("[INFO] Loading YOLO model...")
        self.model = YOLO(self.weights)

        print("[INFO] Starting training...")
        start_time = datetime.datetime.now()

        os.makedirs(log_dir, exist_ok=True)

        # Auto-select GPU if available
        device = 0 if torch.cuda.is_available() else 'cpu'

        # Train with stronger augmentations to improve robustness
        self.model.train(
            data=self.data_path,         # Path to data.yaml
            epochs=self.num_epochs,
            imgsz=self.img_size,         # match dataset resolution multiples
            batch=16,
            device=device,
            workers=4,
            optimizer='ADAM',
            lr0=0.01,
            weight_decay=0.0005,
            patience=30,
            save_period = 10,
            # --- Augmentations ---
            augment=False,
            # degrees=25,                  # random rotation
            # translate=0.15,              # random translation
            # scale=0.6,                   # scaling (zoom in/out)
            # shear=0.0,
            # perspective=0.0,

            # fliplr=0.5,                  # horizontal flip
            # flipud=0.2,                  # vertical flip

            # hsv_h=0.08,                  # hue shift (simulate background variations)
            # hsv_s=0.7,                   # saturation shift
            # hsv_v=0.7,                   # brightness/contrast

            # mosaic=0.6,                  # reduced mosaic to preserve fine details
            # mixup=0.15,                  # light mixup
            # erasing=0.3,                 # random erasing / cutout (if supported in your Ultralytics version)

            project=log_dir,
            name='exp_variety_robust'
        )


        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"[INFO] Training completed in {total_time}")

        # Save trained weights
        output_dir = './weights'
        os.makedirs(output_dir, exist_ok=True)

        trained_model_path = os.path.join('runs', 'detect', 'exp_variety_robust', 'weights', 'best.pt')
        return trained_model_path


if __name__ == "__main__":
    # Paths
    DATA_YAML_PATH = './yolo_dataset1/data.yaml'  # replace with your dataset config
    INITIAL_WEIGHTS = 'yolov8s.pt'               # start from YOLOv8n, can change to yolov8s.pt etc.

    # Train model
    trainer = Training(data_path=DATA_YAML_PATH, yolo_weights=INITIAL_WEIGHTS, num_epochs=150, img_size=640)
    trained_model = trainer.train()


