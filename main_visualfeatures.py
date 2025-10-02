import os
import cv2
import json
import torch
import easyocr
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect import FrameTimecode
from ultralytics import YOLO  # YOLOv8
from fer import FER  # FER+ for emotion
from tqdm import tqdm  # progress bar

# -----------------------
# Parameters
# -----------------------
VIDEO_PATH = "Superman (2025) 4K - Superman vs. The Engineer & Ultraman _ Movieclips.mp4"
OUTPUT_DIR = "analysis_output_VF_realtime_video"
MAX_CHUNK_SEC = 20
SCENE_THRESHOLD = 30.0
DOWNSCALE_FACTOR = 2   # 1 = original res, 2 = half res, 4 = quarter res
MAX_TEST_SECONDS = None  # None = process full video

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Initialize Models
# -----------------------
print("Loading models...")
yolo_model = YOLO("yolov8n.pt")  # small YOLOv8 for objects
ocr_reader = easyocr.Reader(['en'])
emotion_detector = FER(mtcnn=True)

# Placeholder for scene classification
def classify_scene(frame):
    # Replace with Places365 or other classifier later
    return "unknown"

# -----------------------
# Scene Detection
# -----------------------
def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    video_manager.set_downscale_factor(DOWNSCALE_FACTOR)

    # Limit to first N seconds if MAX_TEST_SECONDS is set
    if MAX_TEST_SECONDS is not None:
        video_manager.set_duration(
            start_time=FrameTimecode(0, video_manager.get_framerate()),
            end_time=FrameTimecode(MAX_TEST_SECONDS, video_manager.get_framerate())
        )

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    print("[INFO] Running scene detection...")
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)

    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    print(f"[INFO] Found {len(scene_list)} scenes")
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]   

# -----------------------
# Split chunks
# -----------------------
def split_chunks(scene_list, max_len=MAX_CHUNK_SEC):
    chunks = []
    for start, end in scene_list:
        cur = start
        while cur < end:
            nxt = min(cur + max_len, end)
            chunks.append((cur, nxt))
            cur = nxt
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks

# -----------------------
# Extract keyframe
# -----------------------
def extract_keyframe(video_path, start_time, end_time, output_dir, scene_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_time = (start_time + end_time) / 2
    mid_frame = int(mid_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    if ret:
        out_path = os.path.join(output_dir, f"{scene_id}_keyframe.jpg")
        cv2.imwrite(out_path, frame)
        cap.release()
        return out_path, frame
    cap.release()
    return None, None

# -----------------------
# Analyze Keyframe
# -----------------------
def analyze_keyframe(frame):
    features = {}

    # ---- Objects & Brands ----
    results = yolo_model(frame)
    objects = []
    brands = []
    for r in results:
        for box in r.boxes:
            cls = yolo_model.names[int(box.cls)]
            objects.append(cls)
            # Example brand detection placeholder
            if cls in ["laptop", "phone", "shoe"]:  # dummy mapping
                brands.append("Apple" if cls == "laptop" else "Nike")
    features["objects"] = list(set(objects))
    features["brands"] = list(set(brands))

    # ---- Scene ----
    features["setting"] = classify_scene(frame)

    # ---- On-Screen Text ----
    ocr_result = ocr_reader.readtext(frame)
    features["on_screen_text"] = [text[1] for text in ocr_result]

    # ---- Scene Emotion ----
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        dominant = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        features["scene_emotion"] = dominant
    else:
        features["scene_emotion"] = "neutral"

    return features

# -----------------------
# Build Timeline with Visual Features
# -----------------------
def build_timeline(video_path, chunks, output_dir):
    timeline = []
    for i, (start, end) in enumerate(tqdm(chunks, desc="Analyzing chunks")):
        scene_id = f"scene_{i:03d}"
        keyframe_path, frame = extract_keyframe(video_path, start, end, output_dir, scene_id)

        scene_data = {
            "scene_id": scene_id,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "keyframe": keyframe_path
        }

        if frame is not None:
            visual_features = analyze_keyframe(frame)
            scene_data.update(visual_features)

        timeline.append(scene_data)

    json_path = os.path.join(output_dir, "timeline_visualfeatures.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)

    return json_path

# -----------------------
# Run Pipeline
# -----------------------
if __name__ == "__main__":
    print("Detecting scenes...")
    scenes = detect_scenes(VIDEO_PATH, threshold=SCENE_THRESHOLD)
    chunks = split_chunks(scenes, max_len=MAX_CHUNK_SEC)
    print("Analyzing visual features...")
    json_out = build_timeline(VIDEO_PATH, chunks, OUTPUT_DIR)
    print(f"✅ Visual feature extraction complete! Timeline saved to: {json_out}")
