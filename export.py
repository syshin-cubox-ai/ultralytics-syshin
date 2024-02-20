from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/last.pt")

model.export(format="onnx", simplify=True)
model.export(format="openvino", half=True)
# model.export(format="openvino", int8=True, data="camera.yaml")
