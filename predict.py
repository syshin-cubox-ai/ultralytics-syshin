from ultralytics import YOLO

# model = YOLO("yolov8c-pose.yaml")
# model = YOLO("runs/pose/train/weights/last.pt")
# model = YOLO("runs/pose/train/weights/last.onnx")
# model = YOLO("runs/pose/train/weights/last_openvino_model")
model = YOLO("runs/pose/train/weights/last_int8_openvino_model")
# print(model.info())

results = model.predict(0, show=True, device="cpu", line_width=2)
# results = model.predict("C:/Users/synml/Desktop/images", save=True, device="cpu", line_width=2)
# results = model.predict("ultralytics/assets/largest_selfie.jpg", save=True, line_width=2)
