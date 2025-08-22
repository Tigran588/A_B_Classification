from ultralytics import YOLO
import  cv2
import argparse
import supervision as sv
import numpy as np


class Detection:
    def __init__(self,model_path):
        self.model_path = model_path

    def parse_arguments(self)-> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Yolov8 live')
        parser.add_argument(
            '--webcam-resolution',
            default = [1280,720],
            nargs = 2,
            type = int
        )
        
        args = parser.parse_args()
        return args


    def object_tracking(self):
        ZONE_POLYGON = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])

        args = self.parse_arguments()
        frame_width,frame_height = args.webcam_resolution
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

        model = YOLO(self.model_path)

        box_annotator = sv.BoxAnnotator(
            thickness = 2,
            text_tickness = 2,
            text_scale = 1
        )

        zone_polygon = (ZONE_POLYGON*np.array(args.webcam_resolution)).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution = tuple(args.webcam_resolution))
        zone_annotator = sv.PolygonAnnotator(
            zone = zone,
            color = sv.Color.red(),
            thickness = 2,
            text_thickness = 4,
            text_scale = 2
        )

        if not cap.isOpened():
            print('Could not open Video ')
            exit()

        while True:
            ret,frame = cap.read()
            if not ret:
                print('Could nor red frame')
                break
            
            results = model.track(
                source = frame,
                persist = True,
                conf = 0.5,
                iou =0.5,
                tracker = 'bytetrack.yaml' 
            )

            detections = sv.Detections.from_yolov8(results[0])
            labels = [
                f'{model.model.names[class_id]}{confidence:0.2f}'
                for _,confidence,class_id
                in detections
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels = labels
            )

            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)

            cv2.imshow('yolov8',frame)

            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()     

d = Detection('a')
d.object_tracking()