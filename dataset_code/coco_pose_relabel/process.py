import json
import pathlib
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm

from ultralytics.data.converter import coco91_to_coco80_class
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.files import increment_path

root = pathlib.Path("D:/data/coco-pose-relabel").resolve()
labels_dir = root.parent.joinpath("annotations")


def process_labels(use_keypoints=True, cls91to80=True):
    # Create dataset directory
    save_dir = increment_path(root)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "train" if json_file.stem.find("train") != -1 else "val"
        fn = Path(save_dir) / "labels" / lname  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        img_to_anns = defaultdict(list)
        for ann in data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = img["file_name"]

            bboxes = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                if np.count_nonzero(np.array(ann["keypoints"])) == 0:
                    continue

                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_keypoints and ann.get("keypoints") is not None:
                        if ann.get("face_valid") is not None:
                            if ann["face_valid"]:
                                kpts = np.array(ann["keypoints"], np.float64).reshape(-1, 3)
                                face_kpts = np.array(ann["face_kpts"], np.float64).reshape(-1, 3)

                                kpt_0 = face_kpts[30] if face_kpts[30, 2] > 0 else kpts[0]  # nose
                                if kpts[1, 2] > 0:  # right eye
                                    kpt_1 = kpts[1]
                                else:
                                    if face_kpts[42:48][face_kpts[42:48][..., 2] > 0].size != 0:
                                        kpt_1 = face_kpts[42:48][face_kpts[42:48][..., 2] > 0].mean(0)
                                    else:
                                        kpt_1 = np.zeros_like(kpts[1])
                                if kpts[2, 2] > 0:  # left eye
                                    kpt_2 = kpts[2]
                                else:
                                    if face_kpts[36:42][face_kpts[36:42][..., 2] > 0].size != 0:
                                        kpt_2 = face_kpts[36:42][face_kpts[36:42][..., 2] > 0].mean(0)
                                    else:
                                        kpt_2 = np.zeros_like(kpts[2])
                                kpt_3 = face_kpts[64] if face_kpts[64, 2] > 0 else np.zeros_like(kpts[3])  # right mouth
                                kpt_4 = face_kpts[60] if face_kpts[60, 2] > 0 else np.zeros_like(kpts[4])  # left mouth
                                new_kpts = np.concatenate((kpt_2, kpt_1, kpt_0, kpt_4, kpt_3))
                            else:
                                new_kpts = np.array(ann["keypoints"], np.float64)[:15]
                                new_kpts[9:15] = 0.0  # right mouth, left mouth
                            new_kpts[2:15:3][new_kpts[2:15:3] > 0] = 2.0  # Set visibility 1 to 2
                            new_kpts[0::3] = np.clip(new_kpts[0::3], 0, w)
                            new_kpts[1::3] = np.clip(new_kpts[1::3], 0, h)
                        else:
                            raise KeyError
                            # new_kpts = np.array(ann["keypoints"], np.float64)

                        keypoints.append(
                            box + (new_kpts.reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            # Write
            assert len(bboxes) == len(keypoints)
            if len(bboxes) > 0:
                with open((fn / f).with_suffix(".txt"), "a", newline="\n") as file:
                    for i in range(len(bboxes)):
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                        file.write(f"{line[0]} " + ("%.6f " * len(line[1:])).rstrip() % line[1:] + "\n")

    LOGGER.info(f"COCO data converted successfully.\nResults saved to {save_dir.resolve()}")


def process_images():
    # Extract all
    archive_files = ["train2017.zip", "val2017.zip"]
    for archive_file in archive_files:
        print(f"{archive_file} 압축 푸는 중...")
        with zipfile.ZipFile(root.parent.joinpath(archive_file)) as f:
            f.extractall(root.joinpath("images"))
        print(f"{archive_file} 압축 풀기 완료.")

    # Read paths file
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])

    # Make directories
    train_dir = root.joinpath("images", "train")
    val_dir = root.joinpath("images", "val")
    train_dir.mkdir()
    val_dir.mkdir()

    # Move images
    for label_path in tqdm.tqdm(label_paths, "Move images"):
        dst_path = root.joinpath("images", *label_path.with_suffix(".jpg").parts[-2:])
        parts = list(dst_path.parts)
        parts[-2] += "2017"
        src_path = pathlib.Path().joinpath(*parts)
        src_path.rename(dst_path)


def remove_useless_files():
    shutil.rmtree(root.joinpath("images", "train2017"))
    shutil.rmtree(root.joinpath("images", "val2017"))


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
