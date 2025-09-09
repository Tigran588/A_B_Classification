import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset; replace with custom model if needed)
#model = YOLO('./logs/tensorboard_20250904-100128/exp_variety_robust/weights/best.pt')  # Use 'yolov8s.pt' or other variants for better accuracy
model = YOLO('logs/tensorboard_20250902-121601/exp_variety_robust/weights/best.pt')
# Configure RealSense pipeline for RGB stream (add depth if needed)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB stream at 640x480, 30 FPS
# Optionally add depth: config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames (color and optionally depth)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array (BGR format for OpenCV)
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO inference on the frame
        results = model(color_image, stream=True)  # Use stream=True for real-time efficiency

        # Annotate the frame with bounding boxes, labels, and confidence scores
        annotated_frame = next(results).plot()  # Plot the detections on the frame

        # Display the annotated frame
        cv2.imshow('Real-Time YOLO Inference with RealSense', annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and clean up
    pipeline.stop()
    cv2.destroyAllWindows()