from ultralytics import YOLO

# Load PyTorch YOLOv8 model
model = YOLO("yolov8s.pt")  # download yolov8s.pt if you don't have it

# Export to ONNX
model.export(format="onnx", opset=11)  # creates yolov8s.onnx
