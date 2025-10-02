import os
import cv2
import json
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -----------------------
# Parameters
# -----------------------
VIDEO_PATH = "420.mkv"
OUTPUT_DIR = "analysis_output"
MAX_CHUNK_SEC = 20  # max length before splitting

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Step 1: Scene Detection
# -----------------------
def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    # Returns list of (start_time, end_time) timecodes
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]

# -----------------------
# Step 2: Chunk Splitting
# -----------------------
def split_chunks(scene_list, max_len=MAX_CHUNK_SEC):
    chunks = []
    for start, end in scene_list:
        cur = start
        while cur < end:
            nxt = min(cur + max_len, end)
            chunks.append((cur, nxt))
            cur = nxt
    return chunks

# -----------------------
# Step 3: Keyframe Extraction
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
        return out_path
    cap.release()
    return None

# -----------------------
# Step 4: Build JSON Timeline
# -----------------------
def build_timeline(video_path, chunks, output_dir):
    timeline = []
    for i, (start, end) in enumerate(chunks):
        scene_id = f"scene_{i:03d}"
        keyframe_path = extract_keyframe(video_path, start, end, output_dir, scene_id)

        scene_data = {
            "scene_id": scene_id,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "keyframe": keyframe_path
            # TODO: later add: objects, captions, transcript, audio features
        }
        timeline.append(scene_data)

    json_path = os.path.join(output_dir, "timeline.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)

    return json_path

# -----------------------
# Run Pipeline
# -----------------------
if __name__ == "__main__":
    scenes = detect_scenes(VIDEO_PATH, threshold=30.0)
    chunks = split_chunks(scenes, max_len=MAX_CHUNK_SEC)
    json_out = build_timeline(VIDEO_PATH, chunks, OUTPUT_DIR)

    print(f"✅ Processing complete! Timeline saved to: {json_out}")
