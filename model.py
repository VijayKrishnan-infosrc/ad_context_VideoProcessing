'''from ultralytics import YOLO

# Load PyTorch YOLOv8 model
model = YOLO("yolov8s.pt")  # download yolov8s.pt if you don't have it

# Export to ONNX
model.export(format="onnx", opset=11)  # creates yolov8s.onnx
'''
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # download YOLOv8m PyTorch model if not present
model.export(format="onnx")  # creates yolov8m.onnx
