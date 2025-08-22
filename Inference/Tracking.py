import cv2 
import pyrealsense2 as rs
import numpy  as np
import onnxruntime as ort
import supervision as sv
import logging
import os
import json

from supervision  import Detections
from supervision.tracker.byte_tracker import ByteTrack

logging.basicConfig(level = logging.INFO,format = 'format="%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ONNXDetection:
    def __init__(self,model_path:str,conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_name = None
        self.img_size = None
        self.tracker = ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thicknes=2,text_thickness=2,text_scale = 1)
        self._load_model

    def _load_model(self):
        if not self.model_path.endswith('.onnx') or not os.path.exists(self.model_path):
            logger.error(f'invalid Onnx model path: {self.model_path}')
            exit()
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) != 4 or input_shape !=3:
                raise ValueError(f'unexpected input shape: {input_shape}')
            self.img_size = input_shape[2],input_shape[3]
            logger.info(f'Model loaded {self.model_path} with unput shape {self.img_size}')
        except Exception as e:
            logger.error(f'Failed to load ONNX model: {e}' )
            exit()
        
    def preprocess(self,frame:np.array)->np.array:
            img = cv2.resize(frame,self.img_size)
            img  = img[:,:,::-1] #convert BGR to RGB
            img - img.astype(np.float32)/255.0
            img = np.transpose(img,(2,0,1)) #HWC to CHW
            img = np.expand_dims(img,axis = 0) # add batch dimension
            return img

    def postproces(self,output,orig_shape):
            predictions = output[0][0]
            boxes = []
            for pred in predictions:
                conf = pred[4]
                if conf > self.conf_threshold:
                    x1,y1,x2,y2 = pred[:4]
                    class_id = int(pred[5])
                    x1 = int(x1/self.img_size[1]*orig_shape[1])
                    x2 = int(x2/self.img_size[1]*orig_shape[1])
                    y1 = int(y1/self.img_size[0]*orig_shape[0])
                    y2 = int(y2/self.img_size[0]*orig_shape[0])
                    boxes.append((x1,y1,x2,y2,conf,class_id))
            return boxes
    
    def save_outputs(self,output_data,output_path = 'tracking_outputs.json'):
         try:
              with open(output_path, "w") as f:
                   json.dump(output_data,f,indent=4)
              logger.info(f'outputs saved to {output_path}')
         except Exception as e:
              logger.error
    
    def run(self):
         pipeline = rs.pipeline()
         config = rs.config()
         config.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
         pipeline.start(config)

         output_data = []

         try:
            frame_count = 0
            while True:
                 frames = pipeline.wait_for_frames()
                 color_frame = frames.got_color_frame()
                 if not color_frame:
                      continue
                 
                 frame = np.asanyarray(color_frame)
                 input_tensor = self.preprocess(frame)
                 outputs = self.session.run(None,{self.input_name:input_tensor})
                 boxes = self.postproces(outputs,frame.shape)

                 detections = Detections(
                      xyxy = np.array([[x1,y1,x2,y2] for x1,y1,x2,y2,_,_ in boxes]),
                      confidence = np.array([conf for _,_,_,_,conf,_ in boxes]),
                      class_id = np.array([cls for _,_,_,_,_,cls in boxes])
                 )

                 tracked_detections = self.tracker.update_with_detections(detections)
                 labels = [
                      f'ID:{id} Cls:{cls} {conf:.2f}'
                      for id,(_,_,_,_,conf,cls) in zip(tracked_detections.tracker_id,boxes)
                 ]

                 frame_data = {
                      'frame':frame_count,
                      'objects': [
                        {   
                           '{track_id':int(id),
                           "bbox":[float(x1), float(y1), float(x2), float(y2)],
                           "class": int(cls),
                           "probability": float(conf)
                        }
                     for id, (x1, y1, x2, y2, conf, cls) in zip(tracked_detections.tracker_id, boxes)
                     ]
                 }
                 output_data.append(frame_data)
                 frame_count +=1

                 frame = self.box_annotator.annotate(
                      scene=frame,
                      detections = tracked_detections,
                      labels = labels
                 )

                 cv2.imshow('OBJ. detection', frame )
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                      break
         finally:
              self.save_outputs(output_data)
              pipeline.stop()
              cv2.destroyAllWindows()

if __name__ == '__main__':
     model_path = 'onnx_model/yolo_classification.onnx'
     detector = ONNXDetection(model_path)
     detector.run()
