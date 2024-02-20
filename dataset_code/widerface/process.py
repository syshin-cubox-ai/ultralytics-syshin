import pathlib
import shutil
import zipfile
from functools import reduce

import numpy as np
import tqdm
from PIL import Image

root = pathlib.Path("D:/data/widerface").resolve()


def process_images():
    # Extract image files
    print("이미지 파일 압축 푸는 중...")
    with zipfile.ZipFile(root.parent.joinpath("WIDER_train.zip")) as f:
        f.extractall(root)
    with zipfile.ZipFile(root.parent.joinpath("WIDER_val.zip")) as f:
        f.extractall(root)
    print("이미지 파일 압축 풀기 완료.")

    # Make directories
    train_dir = root.joinpath("images", "train")
    val_dir = root.joinpath("images", "val")
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Move images
    train_img_paths = sorted([i for i in root.joinpath("WIDER_train").rglob("**/*.jpg")])
    val_img_paths = sorted([i for i in root.joinpath("WIDER_val").rglob("**/*.jpg")])
    for train_img_path in tqdm.tqdm(train_img_paths, "Move train images"):
        shutil.move(root.joinpath(train_img_path), train_dir)
    for val_img_path in tqdm.tqdm(val_img_paths, "Move val images"):
        shutil.move(root.joinpath(val_img_path), val_dir)

    # Remove useless directories
    shutil.rmtree(root.joinpath("WIDER_train"))
    shutil.rmtree(root.joinpath("WIDER_val"))


def process_labels():
    # Extract label files
    with zipfile.ZipFile(root.parent.joinpath("retinaface_gt_v1.1.zip")) as f:
        f.extractall(root)
    root.joinpath("train", "label.txt").rename(root.joinpath("train_label.txt"))
    root.joinpath("val", "label.txt").rename(root.joinpath("val_label.txt"))
    shutil.rmtree(root.joinpath("train"))
    shutil.rmtree(root.joinpath("val"))
    shutil.rmtree(root.joinpath("test"))

    # Make directories
    train_dir = root.joinpath("labels", "train")
    val_dir = root.joinpath("labels", "val")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Read labels
    train_labels = {}
    val_labels = {}
    with open(root.joinpath("train_label.txt")) as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:
            if line.startswith("#"):
                file_name = line.rsplit("/")[-1]
                train_labels[file_name] = []
            else:
                label = [float(i) for i in line.split(' ')]
                train_labels[file_name].append(label)
    with open(root.joinpath("val_label.txt")) as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:
            if line.startswith("#"):
                file_name = line.rsplit("/")[-1]
                val_labels[file_name] = []
            else:
                label = [float(i) for i in line.split(' ')]
                val_labels[file_name].append(label)

    # Convert train labels
    debug_file = pathlib.Path("debug.txt")
    debug_file.unlink(missing_ok=True)
    for file_name, label in tqdm.tqdm(train_labels.items(), "Convert train labels"):
        img = Image.open(str(root.joinpath("images", "train", file_name)))
        label = np.array(label)
        with open(train_dir.joinpath(file_name).with_suffix(".txt"), "w") as f:
            for label_one in label:
                # ignore negative value
                if np.count_nonzero(np.where(label_one < 0, 1, 0)) > 0:
                    continue

                label_one = label_one[[0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]]
                label_one[0] = (label_one[0] + label_one[2] / 2) / img.width  # cx
                label_one[1] = (label_one[1] + label_one[3] / 2) / img.height  # cy
                label_one[2] = label_one[2] / img.width  # w
                label_one[3] = label_one[3] / img.height  # h
                label_one[4::2] = label_one[4::2] / img.width
                label_one[5::2] = label_one[5::2] / img.height
                label_one = np.clip(label_one, 0, 1)
                label_one = np.round(label_one, 6)

                f.write(f"0 {' '.join([f'{i:.6f}' for i in label_one])}\n")

    # Convert val labels
    for file_name, label in tqdm.tqdm(val_labels.items(), "Convert val labels"):
        img = Image.open(str(root.joinpath("images", "val", file_name)))
        label = np.array(label)
        with open(val_dir.joinpath(file_name).with_suffix(".txt"), "w") as f:
            for label_one in label:
                label_one[0] = (label_one[0] + label_one[2] / 2) / img.width  # cx
                label_one[1] = (label_one[1] + label_one[3] / 2) / img.height  # cy
                label_one[2] = label_one[2] / img.width  # w
                label_one[3] = label_one[3] / img.height  # h
                label_one = np.clip(label_one, 0, 1)
                label_one = np.round(label_one, 6)

                kpt = (" 0.000000" * reduce(lambda x, y: x * y, [5, 2]))
                f.write(f"0 {' '.join([f'{i:.6f}' for i in label_one])}{kpt}\n")

    # Remove useless files
    root.joinpath("train_label.txt").unlink()
    root.joinpath("val_label.txt").unlink()


def main():
    if root.exists():
        print("파일/폴더가 존재하여 삭제하고 진행합니다.")
        shutil.rmtree(root)
    process_images()
    print("이미지 처리 완료.")
    process_labels()
    print("라벨 처리 완료.")
    print("데이터셋 처리 완료.")


if __name__ == '__main__':
    main()
