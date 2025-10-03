import os
import cv2
import json
import numpy as np
import subprocess
from tqdm import tqdm
import pytesseract

# -----------------------
# Parameters
# -----------------------
VIDEO_PATH = "Superman (2025) 4K - Superman vs. The Engineer & Ultraman _ Movieclips.mp4"
OUTPUT_DIR = "analysis_output_ffmpeg_v2"
SCENE_THRESHOLD = 0.35  # ffmpeg scene detect threshold
MIN_SCENE_LENGTH = 3.0  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Tesseract OCR setup
# -----------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------
# YOLO ONNX Setup (OpenCV DNN)
# -----------------------
YOLO_ONNX_PATH = "yolov8s.onnx"
yolo_net = cv2.dnn.readNetFromONNX(YOLO_ONNX_PATH)

yolo_classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


# -----------------------
# Helper Functions
# -----------------------
def run_ffmpeg_scene_detect(video_path, threshold=0.35):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-filter_complex", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    times = []
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            try:
                ts = float(line.split("pts_time:")[1].split()[0])
                times.append(ts)
            except:
                continue
    # build (start,end) segments
    scenes = []
    last = 0.0
    for t in times:
        if (t - last) >= MIN_SCENE_LENGTH:
            scenes.append((last, t))
        last = t
    return scenes

def extract_keyframes_ffmpeg(video_path, timestamps, scene_id, output_dir):
    paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(output_dir, f"{scene_id}_kf_{i}.jpg")
        cmd = [
            "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
            "-frames:v", "1", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(out_path):
            frame = cv2.imread(out_path)
            paths.append((out_path, frame))
    return paths

def yolo_postprocess(outputs, conf_thres=0.3, iou_thres=0.45):
    boxes, confidences, class_ids = [], [], []
    preds = np.squeeze(outputs).T
    rows = preds.shape[0]
    img_w, img_h = 640, 640
    x_factor, y_factor = 1.0, 1.0

    for i in range(rows):
        row = preds[i]
        confidence = row[4]
        if confidence >= conf_thres:
            scores = row[5:]
            cls_id = np.argmax(scores)
            if scores[cls_id] > conf_thres:
                cx, cy, w, h = row[:4]
                x1 = int((cx - w/2) * x_factor)
                y1 = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    result = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        box = boxes[i]
        cls_id = class_ids[i]
        result.append((cls_id, confidences[i], box))
    return result

def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
    return th

def classify_scene_semantic(objects, motion, text):
    """Simple ad-placement semantic classifier"""
    objs = set(objects)
    if "car" in objs or "bus" in objs or "train" in objs:
        return "vehicle/action"
    elif "person" in objs:
        if motion > 7.0:
            return "action/chase"
        elif len(text) > 0:
            return "dialogue/subtitles"
        else:
            return "dialogue"
    elif "diningtable" in objs or "bottle" in objs:
        return "dining/interior"
    elif motion < 2.0:
        return "static/establishing"
    return "unknown"

def calculate_motion(video_path, start, end, step=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_scores = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if t > end: break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.mean(np.linalg.norm(flow, axis=2))
            motion_scores.append(mag)
            prev_gray = gray
        frame_idx += 1
    cap.release()
    return float(np.mean(motion_scores)) if motion_scores else 0.0

def analyze_keyframes(keyframes):
    objects = set()
    textset = set()
    for path, frame in keyframes:
        # YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward()
        detections = yolo_postprocess(outputs)
        for cls_id, conf, box in detections:
            if cls_id < len(yolo_classes):
                objects.add(yolo_classes[cls_id])
        # OCR
        proc = preprocess_for_ocr(frame)
        text = pytesseract.image_to_string(proc)
        for line in text.splitlines():
            if line.strip():
                textset.add(line.strip())
    return list(objects), list(textset)

# -----------------------
# Main Pipeline
# -----------------------
def build_timeline(video_path, output_dir):
    print("[INFO] Detecting scenes via FFmpeg...")
    scenes = run_ffmpeg_scene_detect(video_path, SCENE_THRESHOLD)
    print(f"[INFO] {len(scenes)} scenes detected")

    timeline = []
    for i, (start, end) in enumerate(tqdm(scenes, desc="Analyzing scenes")):
        scene_id = f"scene_{i:03d}"
        mid = (start+end)/2
        keyframes = extract_keyframes_ffmpeg(video_path, [start, mid, end], scene_id, output_dir)
        objects, text = analyze_keyframes(keyframes)
        motion = calculate_motion(video_path, start, end)
        scene_type = classify_scene_semantic(objects, motion, text)

        timeline.append({
            "scene_id": scene_id,
            "start_time": round(start,2),
            "end_time": round(end,2),
            "duration": round(end-start,2),
            "keyframes": [k[0] for k in keyframes],
            "motion_level": round(motion,2),
            "objects": objects,
            "on_screen_text": text,
            "scene_type": scene_type
        })

    json_path = os.path.join(output_dir, "timeline_hybrid.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"[INFO] JSON saved: {json_path}")
    return json_path

if __name__ == "__main__":
    build_timeline(VIDEO_PATH, OUTPUT_DIR)
