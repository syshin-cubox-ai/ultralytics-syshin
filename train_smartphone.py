import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    opt = parser.parse_args()

    model = YOLO("yolov10n.pt")

    model.train(data="smartphone.yaml", epochs=opt.epochs, imgsz=640, batch=128, optimizer="SGD",
                device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8, plots=True)
