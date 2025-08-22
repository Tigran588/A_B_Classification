from ultralytics import YOLO
import datetime
import os

class Training:
    def __init__(self, data_path, yolo_weights='yolov8n.pt', num_epochs=30, img_size=640):
        self.weights = yolo_weights
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.img_size = img_size

    def train(self):
        log_dir = 'logs/tensorboard_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        print("[INFO] Loading YOLO model...")
        model = YOLO(self.weights)

        print("[INFO] Starting training...")
        start_time = datetime.datetime.now()

        model.train(
            data=self.data_path,
            epochs=self.num_epochs,
            imgsz=self.img_size,
            name='yolo_classification',
            save=True,
            save_period=10,
            optimizer='Adam',
            augment=True,
            mosaic=0.3,
        )

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"[INFO] Training completed in {total_time}")

        # Save the trained weights
        output_dir = './weights'
        os.makedirs(output_dir, exist_ok=True)
        #model.save(output_dir)

        # Return trained model path for export
        trained_model_path = os.path.join('runs', 'detect', 'yolo_classification', 'weights', 'best.pt')
        return trained_model_path


def convert_to_onnx(weights_path: str, export_path: str = './onnx_model'):
    print("[INFO] Loading trained YOLO model for ONNX export...")
    model = YOLO(weights_path)

    print("[INFO] Exporting to ONNX...")
    try:
        model.export(
            format='onnx',
            dynamic=True,
            simplify=True,
            opset=12,
            imgsz=640,
            device='cpu',
            export_dir=export_path
        )
    except Exception as e:
        print(f'Error Onnx export failed:{e}')
        raise 
    
    print(f"[INFO] Model exported to ONNX format at: {export_path}")


if __name__ == "__main__":
    # Set paths and parameters
    DATA_YAML_PATH = 'path/to/data.yaml'  # <-- Replace with your dataset config
    INITIAL_WEIGHTS = 'yolov8n.pt'        # Or a custom starting checkpoint

    # Train
    trainer = Training(data_path=DATA_YAML_PATH, yolo_weights=INITIAL_WEIGHTS)
    trained_model = trainer.train()

    # Export to ONNX
    convert_to_onnx(trained_model)
