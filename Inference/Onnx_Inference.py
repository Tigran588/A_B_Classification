import cv2
import numpy as np
import argparse
import onnxruntime as ort
import supervision as sv
import logging
import os 

logging.basicConfig(level=logging.INFO,format ="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class ONNXDetection:
    def __init__(self, model_path:str,conf_threshold: float = 0.5):
        #path to onnx file
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_name = None
        self.img_size = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.endswith('.onnx'):
            logger.error(f"Model path must end with .onnx: {self.model_path}")
            exit()
        if not os.path.exists(self.model_path):
            logger.error(f'ONNX model not found: {self.model_path}')
            exit()
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) != 4 or input_shape[1] !=3:
                logger.error(f"Unexpected input shape: {input_shape}. Expected [1, 3, H, W]")
                exit()
            self.img_size = (input_shape[2],input_shape[3])
            logger.info(f"Loaded ONNX model: {self.model_path} with input shape {input_shape}")
        except Exception as e:
            logger.error(f'failde to load ONNX model:{e}')
            exit()
        
    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='YOLOv8 ONNX Live Detection')
        parser.add_argument('--webcam-resolution', default=[1280, 720], nargs=2, type=int)
        parser.add_argument('--webcam-reolution',type=int,default=640)
        parser.add_argument('--input-height',type = int, default=640)
        return parser.parse_args()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX inference."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame")
        try:
            img = cv2.resize(frame, self.img_size)
            img = img[:, :, ::-1]  # BGR to RGB
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW (ONNX expects this format)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

    def postprocess(self, output, orig_shape):
        try:
            #assuming there is a one batch and there is one image
            predictions = output[0][0]  # (num_detections, 6)
            boxes = []
            for pred in predictions:
                conf = pred[4]
                if conf > self.conf_threshold:
                    x1, y1, x2, y2 = pred[:4]
                    class_id = int(pred[5])
                    # Rescale to original frame
                    x1 = int(x1 / self.img_size[1] * orig_shape[1])
                    x2 = int(x2 / self.img_size[1] * orig_shape[1])
                    y1 = int(y1 / self.img_size[0] * orig_shape[0])
                    y2 = int(y2 / self.img_size[0] * orig_shape[0])
                    # Ensure valid coordinates
                    boxes.append((x1, y1, x2, y2, conf, class_id))
            return boxes
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return []
        

    def object_detection(self):
        args = self.parse_arguments()
        frame_width, frame_height = args.webcam_resolution

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam. Ensure it is connected and accessible.")
            exit()
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != frame_width or actual_height != frame_height:
            logger.warning(f"Requested resolution {frame_width}x{frame_height} not supported. Using {actual_width}x{actual_height}")
        
        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

        if not cap.isOpened():
            print('Could not open Video ')
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Could not read frame')
                break

            input_tensor = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            boxes = self.postprocess(outputs, frame.shape)

            detections = sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in boxes]),
                confidence=np.array([conf for _, _, _, _, conf, _ in boxes]),
                class_id=np.array([cls for _, _, _, _, _, cls in boxes])
            )

            labels = [
                f"{cls} {conf:.2f}" for (_, _, _, _, conf, cls) in boxes
            ]

            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            cv2.imshow('YOLOv8 ONNX Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace with path to your .onnx model
    detector = ONNXDetection('onnx_model/yolo_classification.onnx')
    detector.object_detection()
