import os, time, logging
import os, time, logging
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import torch 

# -------- CONFIG (tune for speed/quality) --------
DET_WEIGHTS   = "./best.pt"         # <-- put your trained mushroom DET here
IMG_SIZE_DET  = 512                  # try 416 for more speed, 640 for more recall
DET_CONF_TH   = 0.25
DET_IOU_TH    = 0.45

CLS_ONNX      = "runs/classify/train/weights/best.onnx"  # <-- your A/B classifier
CLS_LABELS    = ["Sort_A", "Sort_B"]
IMG_SIZE_CLS  = 256                 # classifier input size (HxW)
SMOOTH_WIN    = 4                   # temporal smoothing window (per box)

DETECT_EVERY_N = 2                  # run detector every N frames (reuse boxes between)

USE_REALSENSE = True                # set False to use default webcam

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mushroom-rt")

# -------------- helpers --------------
def create_onnx_session(path: str) -> ort.InferenceSession:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(path, providers=providers)
    log.info(f"ONNX providers: {sess.get_providers()}")
    return sess

def onnx_input_info(sess: ort.InferenceSession):
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = list(inp.shape)  # e.g. [N,3,256,256] or ['N',3,256,256]
    # treat batch as dynamic if it's None or a string
    dynamic_batch = (shape[0] is None) or isinstance(shape[0], str)
    return name, shape, dynamic_batch

def softmax(logits: np.ndarray) -> np.ndarray:
    # logits shape: (N, num_classes)
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def preprocess_crop_bgr(crop_bgr: np.ndarray, out_hw: int | tuple[int,int]) -> np.ndarray:
    if isinstance(out_hw, int):
        out_hw = (out_hw, out_hw)
    h, w = out_hw
    # letterbox or simple resize; simple resize is fine for square classifiers
    crop = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = crop_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))         # (3,H,W)
    return x

def expand_box(bb_xyxy: np.ndarray, W: int, H: int, scale: float = 1.2):
    x1, y1, x2, y2 = map(float, bb_xyxy)
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    w, h   = (x2-x1), (y2-y1)
    w2, h2 = w*scale, h*scale
    nx1, ny1 = int(max(0, cx - w2/2)), int(max(0, cy - h2/2))
    nx2, ny2 = int(min(W, cx + w2/2)), int(min(H, cy + h2/2))
    return nx1, ny1, nx2, ny2

def draw_box_label(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color=(0,255,0)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y1, th + 6)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

# -------------- video source --------------
def open_stream():
    if USE_REALSENSE:
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        def grab():
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color: 
                return None
            return np.asanyarray(color.get_data())
        def close():
            pipe.stop()
        return grab, close
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        def grab():
            ok, frame = cap.read()
            return frame if ok else None
        def close():
            cap.release()
        return grab, close

# -------------- main --------------
def main():
    log.info(f"Loading detector: {DET_WEIGHTS}")
    det_model = YOLO(DET_WEIGHTS)
    try:
        det_model.to('cuda'); log.info("Detector on CUDA")
    except Exception:
        log.info("Detector on CPU")

    log.info(f"Loading classifier: {CLS_ONNX}")
    cls_sess = create_onnx_session(CLS_ONNX)
    inp_name, inp_shape, dynamic_batch = onnx_input_info(cls_sess)
    out_name = cls_sess.get_outputs()[0].name

    grab, close = open_stream()

    smooth_probs: dict[tuple,int] | dict = {}
    frame_idx = 0
    last_boxes = []
    win = "Mushroom Detect→Classify (A/B)"
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            frame = grab()
            if frame is None:
                continue
            H, W = frame.shape[:2]

            # --- detection every N frames ---
            do_detect = (frame_idx % DETECT_EVERY_N == 0) or (len(last_boxes) == 0)
            if do_detect:
                det_res = det_model.predict(
                    source=frame, imgsz=IMG_SIZE_DET, conf=DET_CONF_TH, iou=DET_IOU_TH,
                    device=0, verbose=False
                )
                boxes = []
                if det_res and len(det_res) > 0 and det_res[0].boxes is not None:
                    b = det_res[0].boxes
                    xyxy = b.xyxy.detach().cpu().numpy()
                    conf = b.conf.detach().cpu().numpy()
                    for i in range(xyxy.shape[0]):
                        boxes.append((xyxy[i], float(conf[i])))
                last_boxes = boxes
            else:
                boxes = last_boxes

            # --- build crops ---
            spans = []
            crops = []
            for bb_xyxy, det_conf in boxes:
                x1,y1,x2,y2 = expand_box(bb_xyxy, W, H, scale=1.2)
                if x2 <= x1 or y2 <= y1: 
                    continue
                crop = frame[y1:y2, x1:x2]
                crops.append(preprocess_crop_bgr(crop, IMG_SIZE_CLS))
                spans.append((x1,y1,x2,y2,det_conf))

            # --- classify (batched if possible) ---
            probs_list = []
            if crops:
                if dynamic_batch and len(crops) > 1:
                    batch = np.stack(crops, axis=0).astype(np.float32)  # (N,3,H,W)
                    logits = cls_sess.run([out_name], {inp_name: batch})[0]
                    probs_list = [p for p in softmax(logits)]
                else:
                    for x in crops:
                        x1 = x[None, ...].astype(np.float32)            # (1,3,H,W)
                        logits = cls_sess.run([out_name], {inp_name: x1})[0]
                        probs_list.append(softmax(logits)[0])

            # --- smoothing + draw ---
            for (x1,y1,x2,y2,detc), p in zip(spans, probs_list):
                key = (x1//12, y1//12, x2//12, y2//12)  # coarse key for temporal assoc
                dq = smooth_probs.get(key)
                if dq is None:
                    dq = deque(maxlen=SMOOTH_WIN)
                    smooth_probs[key] = dq
                dq.append(p)
                p_avg = np.mean(dq, axis=0)

                cls_id = int(np.argmax(p_avg))
                cls_name = CLS_LABELS[cls_id] if cls_id < len(CLS_LABELS) else str(cls_id)
                cls_conf = float(p_avg[cls_id])
                label = f"{cls_name} {cls_conf:.2f} | det {detc:.2f}"
                draw_box_label(frame, x1, y1, x2, y2, label, color=(0,255,0 if cls_id==0 else 255))

            # --- FPS & show ---
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else 1.0/dt
            cv2.putText(frame, f"{fps:.1f} FPS  (det every {DETECT_EVERY_N}f)  {IMG_SIZE_DET=}  {IMG_SIZE_CLS=}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

            frame_idx += 1
    finally:
        close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from collections import deque
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# -------- CONFIG (tune for speed/quality) --------
DET_WEIGHTS   = "best.pt"         # <-- put your trained mushroom DET here
IMG_SIZE_DET  = 512                  # try 416 for more speed, 640 for more recall
DET_CONF_TH   = 0.25
DET_IOU_TH    = 0.45

CLS_ONNX      = "runs/classify/train/weights/best.onnx"  # <-- your A/B classifier
CLS_LABELS    = ["Sort_A", "Sort_B"]
IMG_SIZE_CLS  = 256                 # classifier input size (HxW)
SMOOTH_WIN    = 4                   # temporal smoothing window (per box)

DETECT_EVERY_N = 2                  # run detector every N frames (reuse boxes between)

USE_REALSENSE = True                # set False to use default webcam

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mushroom-rt")

# -------------- helpers --------------
def create_onnx_session(path: str) -> ort.InferenceSession:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(path, providers=providers)
    log.info(f"ONNX providers: {sess.get_providers()}")
    return sess

def onnx_input_info(sess: ort.InferenceSession):
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = list(inp.shape)  # e.g. [N,3,256,256] or ['N',3,256,256]
    # treat batch as dynamic if it's None or a string
    dynamic_batch = (shape[0] is None) or isinstance(shape[0], str)
    return name, shape, dynamic_batch

def softmax(logits: np.ndarray) -> np.ndarray:
    # logits shape: (N, num_classes)
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def preprocess_crop_bgr(crop_bgr: np.ndarray, out_hw: int | tuple[int,int]) -> np.ndarray:
    if isinstance(out_hw, int):
        out_hw = (out_hw, out_hw)
    h, w = out_hw
    # letterbox or simple resize; simple resize is fine for square classifiers
    crop = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = crop_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))         # (3,H,W)
    return x

def expand_box(bb_xyxy: np.ndarray, W: int, H: int, scale: float = 1.2):
    x1, y1, x2, y2 = map(float, bb_xyxy)
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    w, h   = (x2-x1), (y2-y1)
    w2, h2 = w*scale, h*scale
    nx1, ny1 = int(max(0, cx - w2/2)), int(max(0, cy - h2/2))
    nx2, ny2 = int(min(W, cx + w2/2)), int(min(H, cy + h2/2))
    return nx1, ny1, nx2, ny2

def draw_box_label(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color=(0,255,0)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y1, th + 6)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

# -------------- video source --------------
def open_stream():
    if USE_REALSENSE:
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        def grab():
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color: 
                return None
            return np.asanyarray(color.get_data())
        def close():
            pipe.stop()
        return grab, close
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        def grab():
            ok, frame = cap.read()
            return frame if ok else None
        def close():
            cap.release()
        return grab, close

# -------------- main --------------
def main():
    log.info(f"Loading detector: {DET_WEIGHTS}")
    det_model = YOLO(DET_WEIGHTS)
    try:
        det_model.to('cuda'); log.info("Detector on CUDA")
    except Exception:
        log.info("Detector on CPU")

    log.info(f"Loading classifier: {CLS_ONNX}")
    cls_sess = create_onnx_session(CLS_ONNX)
    inp_name, inp_shape, dynamic_batch = onnx_input_info(cls_sess)
    out_name = cls_sess.get_outputs()[0].name

    grab, close = open_stream()

    smooth_probs: dict[tuple,int] | dict = {}
    frame_idx = 0
    last_boxes = []
    win = "Mushroom Detect→Classify (A/B)"
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            frame = grab()
            if frame is None:
                continue
            H, W = frame.shape[:2]

            # --- detection every N frames ---
            do_detect = (frame_idx % DETECT_EVERY_N == 0) or (len(last_boxes) == 0)
            if do_detect:
                det_res = det_model.predict(
                    source=frame, imgsz=IMG_SIZE_DET, conf=DET_CONF_TH, iou=DET_IOU_TH,
                    device=0, verbose=False
                )
                boxes = []
                if det_res and len(det_res) > 0 and det_res[0].boxes is not None:
                    b = det_res[0].boxes
                    xyxy = b.xyxy.detach().cpu().numpy()
                    conf = b.conf.detach().cpu().numpy()
                    for i in range(xyxy.shape[0]):
                        boxes.append((xyxy[i], float(conf[i])))
                last_boxes = boxes
            else:
                boxes = last_boxes

            # --- build crops ---
            spans = []
            crops = []
            for bb_xyxy, det_conf in boxes:
                x1,y1,x2,y2 = expand_box(bb_xyxy, W, H, scale=1.2)
                if x2 <= x1 or y2 <= y1: 
                    continue
                crop = frame[y1:y2, x1:x2]
                crops.append(preprocess_crop_bgr(crop, IMG_SIZE_CLS))
                spans.append((x1,y1,x2,y2,det_conf))

            # --- classify (batched if possible) ---
            probs_list = []
            if crops:
                if dynamic_batch and len(crops) > 1:
                    batch = np.stack(crops, axis=0).astype(np.float32)  # (N,3,H,W)
                    logits = cls_sess.run([out_name], {inp_name: batch})[0]
                    probs_list = [p for p in softmax(logits)]
                else:
                    for x in crops:
                        x1 = x[None, ...].astype(np.float32)            # (1,3,H,W)
                        logits = cls_sess.run([out_name], {inp_name: x1})[0]
                        probs_list.append(softmax(logits)[0])

            # --- smoothing + draw ---
            for (x1,y1,x2,y2,detc), p in zip(spans, probs_list):
                key = (x1//12, y1//12, x2//12, y2//12)  # coarse key for temporal assoc
                dq = smooth_probs.get(key)
                if dq is None:
                    dq = deque(maxlen=SMOOTH_WIN)
                    smooth_probs[key] = dq
                dq.append(p)
                p_avg = np.mean(dq, axis=0)

                cls_id = int(np.argmax(p_avg))
                cls_name = CLS_LABELS[cls_id] if cls_id < len(CLS_LABELS) else str(cls_id)
                cls_conf = float(p_avg[cls_id])
                label = f"{cls_name} {cls_conf:.2f} | det {detc:.2f}"
                draw_box_label(frame, x1, y1, x2, y2, label, color=(0,255,0 if cls_id==0 else 255))

            # --- FPS & show ---
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else 1.0/dt
            cv2.putText(frame, f"{fps:.1f} FPS  (det every {DETECT_EVERY_N}f)  {IMG_SIZE_DET=}  {IMG_SIZE_CLS=}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

            frame_idx += 1
    finally:
        close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
