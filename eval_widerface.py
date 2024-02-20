import argparse
import os
import pathlib
import platform
import shutil
import subprocess

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("weights", type=str, help="model.pt path")
opt = parser.parse_args()

val_img_dir = pathlib.Path("../../data/WIDER_val/images").resolve()
save_dir = pathlib.Path("widerface_evaluate/widerface_txt").resolve()

# Initialize
assert val_img_dir.exists(), f"{val_img_dir} does not exist, extract 'WIDER_val.zip'."
shutil.rmtree(save_dir, ignore_errors=True)
save_dir.mkdir()
model = YOLO(opt.weights)

valset_img_paths = sorted([i for i in val_img_dir.rglob("**/*.jpg")])
for img_path in valset_img_paths:
    results = model.predict(source=img_path, conf=0.001)
    face_boxes = results[0].boxes[results[0].boxes.cls == 0]

    save_name = save_dir.joinpath(*img_path.parts[-2:]).with_suffix(".txt")
    save_name.parent.mkdir(exist_ok=True)
    with open(save_name, "w") as f:
        f.write(f"{save_name.stem}\n")
        f.write(f"{face_boxes.shape[0]}\n")
        for box in face_boxes:
            conf = box.conf[0].clip(0, 1)
            xyxy = box.xyxy[0].round()
            x1 = xyxy[0]
            y1 = xyxy[1]
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            f.write(f"{x1} {y1} {w} {h} {conf:.6f}\n")

if platform.system() == "Windows":
    os.chdir("widerface_evaluate")
    subprocess.run("python evaluation.py")
