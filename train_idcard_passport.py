import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)  # 200
    opt = parser.parse_args()

    model = YOLO("yolov8n-seg.pt")

    model.train(data="idcard-passport-seg.yaml", epochs=opt.epochs, imgsz=640, batch=128, optimizer="SGD",
                device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8, plots=True,
                perspective=0.0001, mask_ratio=1)
