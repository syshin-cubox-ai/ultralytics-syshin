from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")

    model.train(data="face-mix.yaml", epochs=400, imgsz=640, batch=128, optimizer="SGD",
                device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8, plots=True)
