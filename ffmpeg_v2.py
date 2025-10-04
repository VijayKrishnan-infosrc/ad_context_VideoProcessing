import os
import cv2
import json
import numpy as np
import subprocess
from tqdm import tqdm
import pytesseract
from PIL import Image
import torch
import librosa
from deepface import DeepFace
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------------------------------------------------------------
# --- 1. PARAMETERS & SETUP ---
# --------------------------------------------------------------------------

VIDEO_PATH = "Superman (2025) 4K - Superman vs. The Engineer & Ultraman _ Movieclips.mp4"
OUTPUT_DIR = "analysis_output_rich_v3"
SCENE_THRESHOLD = 0.35  # FFmpeg scene detect threshold (0.3 - 0.5 is a good range)
MIN_SCENE_LENGTH = 3.0  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Tesseract OCR Setup ---
# Make sure Tesseract is in your PATH or provide the full path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- YOLO ONNX Setup ---
YOLO_ONNX_PATH = "yolov8m.onnx" # Using the medium model for better accuracy
print(f"[INFO] Loading YOLO model: {YOLO_ONNX_PATH}")
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


# --- Image Captioning Model Setup (loads once) ---
print("[INFO] Loading Image Captioning model (this may take a few minutes on first run)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
print(f"[INFO] Models loaded successfully on device: {device}")


# --------------------------------------------------------------------------
# --- 2. CORE ANALYSIS FUNCTIONS ---
# --------------------------------------------------------------------------

def run_ffmpeg_scene_detect(video_path, threshold):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter_complex", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    times = [0.0] # Start at the beginning
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            try:
                ts = float(line.split("pts_time:")[1].split()[0])
                times.append(ts)
            except: continue
    
    # Get video duration to cap the last scene
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    duration_res = subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True)
    video_duration = float(duration_res.stdout.strip())
    if times[-1] < video_duration:
        times.append(video_duration)

    scenes = []
    for i in range(len(times) - 1):
        start, end = times[i], times[i+1]
        if (end - start) >= MIN_SCENE_LENGTH:
            scenes.append((start, end))
    return scenes

def extract_keyframe_ffmpeg(video_path, timestamp, output_path):
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(output_path):
        return cv2.imread(output_path)
    return None

# --- CRITICAL FIX: Corrected object detection post-processing ---
def yolo_postprocess(outputs, original_shape, conf_thres=0.3, iou_thres=0.45):
    boxes, confidences, class_names = [], [], []
    original_h, original_w = original_shape[:2]
    model_input_w, model_input_h = 640, 640
    x_factor = original_w / model_input_w
    y_factor = original_h / model_input_h
    
    preds = np.squeeze(outputs).T

    for row in preds:
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
                class_names.append(yolo_classes[cls_id])

    if not boxes: return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    if indices is None or len(indices) == 0: return []
        
    return [class_names[i] for i in indices.flatten()]

def analyze_visuals(frame):
    """Analyzes a single frame for objects and text."""
    # --- Object Detection ---
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward()
    objects = yolo_postprocess(outputs, frame.shape)
    
    # --- OCR ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    text_content = set()
    try:
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        for line in text.splitlines():
            clean_line = line.strip()
            if len(clean_line) > 2 and any(char.isalnum() for char in clean_line):
                text_content.add(clean_line)
    except: pass
    
    return list(set(objects)), list(text_content)

def analyze_audio_segment(video_path, start_time, end_time):
    try:
        y, sr = librosa.load(video_path, sr=22050, offset=start_time, duration=end_time-start_time)
        if np.mean(librosa.feature.rms(y=y)) < 0.005: return "silence"
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        if np.mean(np.abs(y_harmonic)) > np.mean(np.abs(y_percussive)):
            return "dialogue/music"
        else:
            return "action/effects"
    except Exception:
        return "unknown"

def analyze_faces_emotions(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        emotions = set()
        if isinstance(results, list): # Multiple faces
            for face in results:
                emotions.add(face.get('dominant_emotion'))
        elif isinstance(results, dict): # Single face
            emotions.add(results.get('dominant_emotion'))
        return [e for e in emotions if e] # Filter out None
    except:
        return []

def generate_caption(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = processor(pil_image, return_tensors="pt").to(device)
        out = caption_model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        return "Caption generation failed."

def calculate_motion(video_path, start, end):
    # This is still slow; for production, a faster method may be needed.
    # For now, it remains as is for consistency.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    ret, prev = cap.read()
    if not ret: return 0.0
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_scores = []
    
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end: break
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.mean(np.linalg.norm(flow, axis=2))
        motion_scores.append(mag)
        prev_gray = gray
        
    cap.release()
    return float(np.mean(motion_scores)) if motion_scores else 0.0

# --------------------------------------------------------------------------
# --- 3. MAIN PIPELINE ORCHESTRATION ---
# --------------------------------------------------------------------------

def build_timeline(video_path, output_dir):
    print("\n[INFO] Step 1: Detecting scenes via FFmpeg...")
    scenes = run_ffmpeg_scene_detect(video_path, SCENE_THRESHOLD)
    print(f"[INFO] Found {len(scenes)} scenes >= {MIN_SCENE_LENGTH}s")

    timeline = []
    print("\n[INFO] Step 2: Analyzing scenes...")
    for i, (start, end) in enumerate(tqdm(scenes, desc="Processing Scenes")):
        scene_id = f"scene_{i:03d}"
        mid_timestamp = (start + end) / 2
        keyframe_path = os.path.join(output_dir, f"{scene_id}_keyframe.jpg")

        keyframe_img = extract_keyframe_ffmpeg(video_path, mid_timestamp, keyframe_path)
        if keyframe_img is None:
            continue

        # --- Run all analyses ---
        objects, text = analyze_visuals(keyframe_img)
        motion = calculate_motion(video_path, start, end)
        audio_type = analyze_audio_segment(video_path, start, end)
        emotions = analyze_faces_emotions(keyframe_img)
        caption = generate_caption(keyframe_img)
        
        timeline.append({
            "scene_id": scene_id,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "keyframe_path": keyframe_path,
            # --- High-Value Attributes ---
            "semantic_caption": caption,
            "audio_type": audio_type,
            "emotions_detected": emotions,
            "objects_detected": objects,
            "motion_level": round(motion, 2),
            "on_screen_text": text
        })

    json_path = os.path.join(output_dir, "timeline_rich_analysis.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\n[SUCCESS] Rich analysis complete. JSON saved to: {json_path}")
    return json_path

if __name__ == "__main__":
    build_timeline(VIDEO_PATH, OUTPUT_DIR)