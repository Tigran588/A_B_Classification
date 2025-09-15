import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
try:
    model = YOLO('./weights/vision_transformers.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB stream
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # Depth stream

# Align depth to color
align = rs.align(rs.stream.color)

# Start streaming
try:
    pipeline.start(config)
except Exception as e:
    print(f"Error starting RealSense pipeline: {e}")
    exit(1)

try:
    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Get depth scale (to convert depth values to meters)
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        max_distance_mm = 300  # 30 cm limit

        # Run YOLO inference
        results = model(color_image, stream=True)

        # Process results
        for result in results:
            annotated_frame = color_image.copy()  # Create a copy of the frame
            boxes = result.boxes  # Get detection boxes

            if len(boxes) > 0:  # Check if there are any detections
                # Get confidence scores
                confidences = boxes.conf.cpu().numpy()
                # Find index of the highest confidence score
                max_conf_idx = np.argmax(confidences)
                max_conf = confidences[max_conf_idx]
                box = boxes[max_conf_idx]

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract depth values within the bounding box
                depth_roi = depth_image[y1:y2, x1:x2]
                # Filter out invalid depth values (0) and convert to millimeters
                valid_depths = depth_roi[depth_roi > 0] * depth_scale * 1000  # Convert to mm
                if len(valid_depths) > 0:
                    avg_depth_mm = np.mean(valid_depths)
                    # Check if the object is within 30 cm
                    if avg_depth_mm <= max_distance_mm:
                        label = result.names[int(box.cls)]  # Class label
                        conf = box.conf.item()  # Confidence score

                        # Draw the bounding box, label, and confidence
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f"{label}: {conf:.2f}, Depth: {avg_depth_mm:.0f}mm"
                        cv2.putText(
                            annotated_frame,
                            label_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            # Display the annotated frame
            cv2.imshow('Real-Time YOLO Inference with RealSense (≤30cm)', annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and clean up
    pipeline.stop()
    cv2.destroyAllWindows()