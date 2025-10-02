# 🎥 Ad Context Video Processing

This project is a **video analysis pipeline** that detects scenes, extracts visual features, and builds a structured timeline for video content.  
It combines **scene detection, object detection (YOLO), OCR (Tesseract), and motion analysis** into one automated workflow.
main_updated_pipeline.py has the recent version of pipeline.
---

## ⚡ Features
- **Scene Detection** → Splits video into meaningful shots using `scenedetect`.
- **Keyframe Extraction** → Saves representative frames (start, middle, end).
- **Object Detection** → Uses **YOLOv8 (ONNX, OpenCV DNN)** trained on **COCO-80 classes**.
- **OCR** → Extracts on-screen text with **Tesseract**.
- **Motion Analysis** → Computes scene motion level via **optical flow**.
- **Timeline Export** → Outputs results as `timeline_visualfeatures.json`.

---

## 📂 Project Structure
ad_context_VideoProcessing/
│── main.py # Entry script
│── main_updated_pipeline.py # Updated full pipeline- latest(visual features only)
│── main_visualfeatures.py # Testing Visual features module
│── model.py # Model helpers
│── requirements.txt # Dependencies
│── .gitignore 
│── analysis_output_* # Output folders
