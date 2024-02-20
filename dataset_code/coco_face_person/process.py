import os
import pathlib
import shutil
import zipfile
from functools import reduce

import tqdm

root = pathlib.Path("D:/data/coco-face-person").resolve()
face_class = 0
person_class = 1


def process_labels():
    print("coco2017labels-pose.zip 압축 푸는 중. . .")
    with zipfile.ZipFile(root.parent.joinpath("coco2017labels-pose.zip")) as f:
        f.extractall(root.parent)
    print("coco2017labels-pose.zip 압축 풀기 완료.")
    root.parent.joinpath("coco-pose").rename(root)
    root.joinpath("labels", "train2017").rename(root.joinpath("labels", "train"))
    root.joinpath("labels", "val2017").rename(root.joinpath("labels", "val"))

    # Process labels
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])
    for label_path in tqdm.tqdm(label_paths, "Process labels"):
        # Person label
        with open(label_path) as f:
            person_label = f.readlines()
        kpt = (" 0.000000" * reduce(lambda x, y: x * y, [5, 2]))
        person_label = [f"{person_class} {' '.join(i.split()[1:5])}{kpt}\n" for i in person_label]

        # Face label
        coco_face_label_path = root.parent.joinpath("coco-face", "/".join(label_path.parts[-3:]))
        if coco_face_label_path.is_file():
            with open(coco_face_label_path) as f:
                face_label = f.readlines()
            face_label = [f"{face_class} {' '.join(i.split()[1:])}\n" for i in face_label]
        else:
            face_label = []

        # Write label
        label = face_label + person_label
        with open(label_path, "w") as f:
            f.writelines(label)


def process_images():
    print("train2017.zip 압축 푸는 중...")
    with zipfile.ZipFile(root.parent.joinpath("train2017.zip")) as f:
        f.extractall(root.joinpath("images"))
    print("train2017.zip 압축 풀기 완료.")
    print("val2017.zip 압축 푸는 중...")
    with zipfile.ZipFile(root.parent.joinpath("val2017.zip")) as f:
        f.extractall(root.joinpath("images"))
    print("val2017.zip 압축 풀기 완료.")

    # Read paths file
    with open(root.joinpath("train2017.txt")) as f:
        train_paths = f.readlines()
    with open(root.joinpath("val2017.txt")) as f:
        val_paths = f.readlines()
    train_paths = sorted([i.strip()[2:] for i in train_paths])
    val_paths = sorted([i.strip()[2:] for i in val_paths])

    # Make directories
    train_dir = root.joinpath("images", "train")
    val_dir = root.joinpath("images", "val")
    train_dir.mkdir()
    val_dir.mkdir()

    # Move images
    for train_path in tqdm.tqdm(train_paths, "Move train images"):
        shutil.move(root.joinpath(train_path), train_dir.joinpath(os.path.basename(train_path)))
    for val_path in tqdm.tqdm(val_paths, "Move val images"):
        shutil.move(root.joinpath(val_path), val_dir.joinpath(os.path.basename(val_path)))


def remove_useless_files():
    shutil.rmtree(root.joinpath("annotations"))
    shutil.rmtree(root.joinpath("images", "train2017"))
    shutil.rmtree(root.joinpath("images", "val2017"))
    root.joinpath("train2017.txt").unlink()
    root.joinpath("val2017.txt").unlink()


def main():
    if root.exists():
        print("파일/폴더가 존재하여 삭제하고 진행합니다.")
        shutil.rmtree(root)
    process_labels()
    print("라벨 처리 완료.")
    process_images()
    print("이미지 처리 완료.")
    remove_useless_files()
    print("데이터셋 처리 완료.")


if __name__ == "__main__":
    main()
