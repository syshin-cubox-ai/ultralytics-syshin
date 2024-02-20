import os
import pathlib
import shutil
import zipfile

import tqdm

root = pathlib.Path("D:/data/coco-face").resolve()
# root = pathlib.Path("/purestorage/project/syshin/data/coco-face").resolve()


def build_dataset_structure():
    os.makedirs(root, exist_ok=True)
    os.makedirs(root.joinpath("images", "train"), exist_ok=True)
    os.makedirs(root.joinpath("images", "val"), exist_ok=True)
    os.makedirs(root.joinpath("labels", "train"), exist_ok=True)
    os.makedirs(root.joinpath("labels", "val"), exist_ok=True)


def prepare_images():
    with zipfile.ZipFile(root.parent.joinpath("train2017.zip")) as f:
        f.extractall(root.joinpath("images"))
    with zipfile.ZipFile(root.parent.joinpath("val2017.zip")) as f:
        f.extractall(root.joinpath("images"))

    # Read paths file
    with open(pathlib.Path(__file__).parent.joinpath("coco_pose_images_train_paths.txt")) as f:
        train_paths = f.readlines()
    with open(pathlib.Path(__file__).parent.joinpath("coco_pose_images_val_paths.txt")) as f:
        val_paths = f.readlines()
    train_paths = sorted([i.strip() for i in train_paths])
    val_paths = sorted([i.strip() for i in val_paths])

    # paths file이 상대경로이므로 현재 디렉토리를 root로 변경
    os.chdir(root)

    # Move images
    train_dir = root.joinpath("images", "train")
    val_dir = root.joinpath("images", "val")
    for train_path in tqdm.tqdm(train_paths, "train images"):
        shutil.move(train_path, train_dir.joinpath(os.path.basename(train_path)))
    for val_path in tqdm.tqdm(val_paths, "val images"):
        shutil.move(val_path, val_dir.joinpath(os.path.basename(val_path)))

    # Remove useless files
    shutil.rmtree(train_dir.parent.joinpath("train2017"))
    shutil.rmtree(val_dir.parent.joinpath("val2017"))


def main():
    build_dataset_structure()
    print("데이터셋 구조 빌드 완료.")
    prepare_images()
    print("데이터셋 이미지 준비 완료.")
    print("pseudo labeling을 수행해주세요.")


if __name__ == "__main__":
    main()
