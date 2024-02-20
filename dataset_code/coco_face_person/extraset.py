import pathlib
import shutil
from functools import reduce

from dataset_code.coco_face_person.process import person_class
from ultralytics import YOLO

# Paths
widerface_root = pathlib.Path("D:/data/widerface").resolve(strict=True)
widerface_train_images_dir = widerface_root.joinpath("images", "train")
widerface_train_labels_dir = widerface_root.joinpath("labels", "train")
extraset_root = pathlib.Path("D:/data/coco-face-person-extra").resolve()
shutil.rmtree(extraset_root, ignore_errors=True)


def main():
    model = YOLO("yolov9e.pt")
    results = model.predict(widerface_train_images_dir, stream=True, classes=[0], conf=0.3)
    kpt = (" 0.000000" * reduce(lambda x, y: x * y, [5, 2]))

    # Pseudo labeling
    for result in results:
        label = [f"{person_class} {i[0]:.6f} {i[1]:.6f} {i[2]:.6f} {i[3]:.6f}{kpt}\n"
                 for i in result.boxes.xywhn]
        label_path = widerface_train_labels_dir.joinpath(pathlib.Path(result.path).with_suffix(".txt").name)
        with open(label_path, "a") as f:
            f.writelines(label)

    # Build dataset structure
    extraset_root.joinpath("images", "train").mkdir(parents=True)
    extraset_root.joinpath("labels", "train").mkdir(parents=True)
    widerface_src_paths = [i for i in widerface_root.joinpath("images", "train").rglob("**/*.*")]
    widerface_src_paths = widerface_src_paths + [i for i in widerface_root.joinpath("labels", "train").rglob("**/*.*")]
    for src_path in widerface_src_paths:
        parts = list(src_path.parts)
        parts[-4] = extraset_root.name
        dest_path = pathlib.Path(*parts)
        src_path.rename(dest_path)

    # Remove useless files
    shutil.rmtree(widerface_root)


if __name__ == "__main__":
    main()
