import cv2 
import pyrealsense2 as rs
import numpy as np
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getlogger(__name__)

class RealTimeClassiFIer:
    def __init__(self,model_path: str, class_names :str):
        self.model_path = model_path
        self.class_names = class_names
        self._load_model()
    
    def _load_model(self):
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.img_size = (input_shape[2]) #[batch,chanels,height,width]
        logger.info(f'Loaded model: {self.model_path} with_input_shape {input_shape}')

    def preprocess(self,frame):
        frame_resized = cv2.resize(frame,self.img_size)
        frame_rgb = cv2.cvtColor(frame_resized,cv2.COLOR_BGR2RGB)
        img = frame_rgb.astype(np.float32)/255.0
        img = np.transpose(img,2,0,1)  
        img = np.expand_dims(img,axis = 0)
        return img 
    
    def classify(self,frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None,{self.input_name:input_tensor})
        probs = outputs[0][0]
        class_id = int(np.argmax(probs))    
        confidence = float(probs[class_id])
        return self.class_names[class_id],confidence
    
    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        pipeline.start(config)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.array(color_frame.get_data())

                label,conf = self.classify(frame)
                cv2.putText(frame, f"{label} ({conf:.2f})", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    class_names = ['Sort_A','Sort_B']
    model_path = ''

    classifier = RealTimeClassiFIer(model_path,class_names)
    classifier.run()
    