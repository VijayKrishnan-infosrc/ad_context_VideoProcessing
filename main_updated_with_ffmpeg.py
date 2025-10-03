import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import pytesseract
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import subprocess # --- NEW IMPORT for FFmpeg ---

# -----------------------
# Parameters
# -----------------------
VIDEO_PATH = "Superman (2025) 4K - Superman vs. The Engineer & Ultraman _ Movieclips.mp4"
OUTPUT_DIR = "analysis_output_ad_optimized_V1"
SCENE_THRESHOLD = 30.0  # Lowered threshold slightly to catch more scenes
MIN_SCENE_LENGTH = 3.0
DOWNSCALE_FACTOR = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Tesseract OCR setup
# -----------------------
# Make sure to have Tesseract installed and in your system's PATH
# or provide the full path as you did.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------
# ONNX YOLO Setup
# -----------------------
YOLO_ONNX_PATH = "yolov8s.onnx"
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

# --- IMPROVEMENT: A more intelligent placeholder scene classifier ---
def classify_scene_type(detected_objects, motion_level):
    """Classifies scene type based on detected objects and motion."""
    objects_set = set(detected_objects)
    if motion_level > 8.0 and ('car' in objects_set or 'person' in objects_set):
        return "action/chase"
    if 'person' in objects_set and len(objects_set) <= 3 and motion_level < 4.0:
        return "dialogue/character-focused"
    if 'diningtable' in objects_set or 'cup' in objects_set or 'bottle' in objects_set:
        return "dining/interior"
    if 'car' in objects_set or 'truck' in objects_set or 'bus' in objects_set:
        return "exterior/vehicle"
    return "unknown"

# -----------------------
# Scene Detection (Unchanged)
# -----------------------
def detect_scenes(video_path, threshold=SCENE_THRESHOLD, min_scene_length=MIN_SCENE_LENGTH):
    video_manager = VideoManager([video_path])
    video_manager.set_downscale_factor(DOWNSCALE_FACTOR)
    video_manager.start()
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    print("[INFO] Running scene detection...")
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    filtered_scenes = [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in scene_list if (s[1].get_seconds() - s[0].get_seconds()) >= min_scene_length
    ]
    print(f"[INFO] {len(filtered_scenes)} scenes detected (>= {min_scene_length}s)")
    return filtered_scenes

# --- FFmpeg INTEGRATION: Faster and more accurate keyframe extraction ---
def extract_keyframe_ffmpeg(video_path, timestamp, output_path):
    """Extracts a single frame from the video using FFmpeg."""
    command = [
        'ffmpeg',
        '-ss', str(timestamp),      # Seek to the exact timestamp
        '-i', video_path,
        '-vframes', '1',            # Extract only one frame
        '-q:v', '2',                # High quality
        '-y',                       # Overwrite output file if it exists
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame = cv2.imread(output_path)
        return frame
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[ERROR] FFmpeg failed for timestamp {timestamp}: {e}")
        return None

# -----------------------
# Motion Level (Unchanged, but consider simplifying if too slow)
# -----------------------
def calculate_motion_level(video_path, start_time, end_time):
    # This function is computationally expensive. For a faster pipeline,
    # you might analyze fewer frames or use a simpler metric.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time*fps))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_scores = []
    frame_count = 0
    while cap.isOpened():
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > end_time:
            break
        ret, frame = cap.read()
        if not ret:
            break
        # Process every Nth frame to speed up
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(np.mean(mag))
            prev_gray = gray
        frame_count += 1
    cap.release()
    return float(np.mean(motion_scores)) if motion_scores else 0.0

# --- REVISED FUNCTION: Corrected YOLO processing and OCR pre-processing ---
def analyze_keyframes(keyframes_data):
    scene_features = {"objects": set(), "on_screen_text": set()}
    
    CONF_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.45
    
    for path, frame in keyframes_data:
        if frame is None:
            continue

        # ---- Object Detection (YOLOv8 ONNX with NMS) ----
        h, w, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())[0]
        
        # The output shape is (84, 8400). Transpose it to (8400, 84).
        # 8400 detections, each with 84 values (4 for box, 80 for class scores)
        outputs = np.squeeze(outputs).T

        boxes, confidences, class_ids = [], [], []

        for det in outputs:
            # First 4 values are box coords, the 5th is confidence, rest are class scores
            confidence = det[4]
            if confidence >= CONF_THRESHOLD:
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                if class_scores[class_id] > 0.25: # Check class score as well
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    # Bbox needs to be scaled back to original image size
                    cx, cy, cw, ch = det[:4]
                    x = int((cx - cw/2) * w)
                    y = int((cy - ch/2) * h)
                    width = int(cw * w)
                    height = int(ch * h)
                    boxes.append([x, y, width, height])

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                class_id = class_ids[i]
                if class_id < len(yolo_classes):
                    scene_features["objects"].add(yolo_classes[class_id])

        # ---- OCR with pre-processing ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Applying a threshold to create a binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            for line in text.splitlines():
                if line.strip() and len(line.strip()) > 2: # Filter very short noise
                    scene_features["on_screen_text"].add(line.strip())
        except Exception as e:
            print(f"[ERROR] Tesseract failed: {e}")

    # Convert sets to lists for JSON serialization
    scene_features["objects"] = list(scene_features["objects"])
    scene_features["on_screen_text"] = list(scene_features["on_screen_text"])
    return scene_features

# -----------------------
# Build Timeline (Modified to use FFmpeg and new classifier)
# -----------------------
def build_scene_timeline(video_path, scenes, output_dir):
    timeline = []
    for i, (start, end) in enumerate(tqdm(scenes, desc="Analyzing scenes")):
        scene_id = f"scene_{i:03d}"
        
        # Extract 3 keyframes: start, middle, end of the scene
        timestamps = [start, (start + end) / 2, end - 0.1]
        keyframes_data = []
        keyframe_paths = []
        
        for idx, t in enumerate(timestamps):
            path = os.path.join(output_dir, f"{scene_id}_keyframe_{idx}.jpg")
            frame = extract_keyframe_ffmpeg(video_path, t, path)
            if frame is not None:
                keyframes_data.append((path, frame))
                keyframe_paths.append(path)

        features = analyze_keyframes(keyframes_data)
        motion_level = calculate_motion_level(video_path, start, end)
        scene_type = classify_scene_type(features["objects"], motion_level)

        timeline.append({
            "scene_id": scene_id,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "keyframes": keyframe_paths,
            "motion_level": round(motion_level, 3),
            "scene_type": scene_type,
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