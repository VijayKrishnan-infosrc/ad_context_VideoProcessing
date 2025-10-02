import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import pytesseract
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -----------------------
# Parameters
# -----------------------
VIDEO_PATH = "Superman (2025) 4K - Superman vs. The Engineer & Ultraman _ Movieclips.mp4"
OUTPUT_DIR = "analysis_output_ad_optimized"
MAX_CHUNK_SEC = 20
SCENE_THRESHOLD = 50.0
MIN_SCENE_LENGTH = 3.0
DOWNSCALE_FACTOR = 2  # reduce for speed, keep 1 for full res

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Tesseract OCR setup
# -----------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust path

# -----------------------
# ONNX YOLO Setup (OpenCV DNN)
# -----------------------
YOLO_ONNX_PATH = "yolov8s.onnx"  # place yolov8s.onnx in your folder
yolo_net = cv2.dnn.readNetFromONNX(YOLO_ONNX_PATH)
yolo_classes = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# -----------------------
# Placeholder Scene Type Classifier
# -----------------------
def classify_scene_type(frame):
    return "unknown"

# -----------------------
# Scene Detection
# -----------------------
def detect_scenes(video_path, threshold=SCENE_THRESHOLD, min_scene_length=MIN_SCENE_LENGTH):
    video_manager = VideoManager([video_path])
    video_manager.set_downscale_factor(DOWNSCALE_FACTOR)
    video_manager.start()

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    print("[INFO] Running scene detection...")
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)

    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    filtered_scenes = [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in scene_list if (s[1].get_seconds() - s[0].get_seconds()) >= min_scene_length
    ]
    print(f"[INFO] {len(filtered_scenes)} scenes detected (>= {min_scene_length}s)")
    return filtered_scenes

# -----------------------
# Extract Keyframes
# -----------------------
def extract_keyframes(video_path, start_time, end_time, output_dir, scene_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    times = [start_time, (start_time + end_time)/2, end_time]
    for idx, t in enumerate(times):
        frame_num = int(t*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, f"{scene_id}_keyframe_{idx}.jpg")
            cv2.imwrite(path, frame)
            keyframes.append((path, frame))
    cap.release()
    return keyframes

# -----------------------
# Motion Level
# -----------------------
def calculate_motion_level(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time*fps))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_scores = []

    while True:
        ret, frame = cap.read()
        if not ret or (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0) > end_time:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        motion_scores.append(np.mean(np.linalg.norm(flow, axis=2)))
        prev_gray = gray

    cap.release()
    return float(np.mean(motion_scores)) if motion_scores else 0.0

# -----------------------
# Analyze Keyframes
# -----------------------
def analyze_keyframes(keyframes):
    scene_features = {
        "objects": set(),
        "on_screen_text": set(),
        "scene_type": set()
    }

    for path, frame in keyframes:
        # ---- Object Detection (ONNX YOLO + OpenCV) ----
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward()

        if len(outputs) > 0:
            for det in outputs[0]:
                det = det.flatten()
                if det.shape[0] < 5:
                    continue

                obj_conf = det[4]
                class_scores = det[5:]
                if len(class_scores) == 0:
                    continue

                cls_id = int(np.argmax(class_scores))
                conf = float(class_scores[cls_id] * obj_conf)

                if conf > 0.3 and cls_id < len(yolo_classes):
                    scene_features["objects"].add(yolo_classes[cls_id])

        # ---- OCR ----
        text = pytesseract.image_to_string(frame)
        for line in text.splitlines():
            if line.strip():
                scene_features["on_screen_text"].add(line.strip())

        # ---- Scene type ----
        stype = classify_scene_type(frame)
        scene_features["scene_type"].add(stype)

    scene_features["objects"] = list(scene_features["objects"])
    scene_features["on_screen_text"] = list(scene_features["on_screen_text"])
    scene_features["scene_type"] = list(scene_features["scene_type"])
    return scene_features

# -----------------------
# Build Timeline
# -----------------------
def build_scene_timeline(video_path, scenes, output_dir):
    timeline = []
    for i, (start, end) in enumerate(tqdm(scenes, desc="Analyzing scenes")):
        scene_id = f"scene_{i:03d}"
        keyframes = extract_keyframes(video_path, start, end, output_dir, scene_id)
        features = analyze_keyframes(keyframes)
        motion_level = calculate_motion_level(video_path, start, end)

        timeline.append({
            "scene_id": scene_id,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end-start, 2),
            "keyframes": [k[0] for k in keyframes],
            "motion_level": round(motion_level, 3),
            **features
        })

    json_path = os.path.join(output_dir, "timeline_visualfeatures.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)

    return json_path

# -----------------------
# Run Pipeline
# -----------------------
if __name__ == "__main__":
    print("[INFO] Detecting scenes...")
    scenes = detect_scenes(VIDEO_PATH)
    print("[INFO] Analyzing scenes and extracting features...")
    json_out = build_scene_timeline(VIDEO_PATH, scenes, OUTPUT_DIR)
    print(f"✅ Ad-analysis JSON saved to: {json_out}")
