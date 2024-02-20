import os
import pathlib
import shutil
import sys

import cv2
import numpy as np
import ray
from ray.experimental import tqdm_ray

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from dataset_code.coco_face.preprocess import root
from dataset_code.coco_face.yolo5face import YOLO5Face, draw_prediction, xyxy2xywhn

remote_tqdm = ray.remote(tqdm_ray.tqdm)


@ray.remote
def worker(img_paths: list[pathlib.Path], gpu_id: int, progress_bar: tqdm_ray.tqdm):
    # no faces threshold: 0.76
    # labeling threshold: 0.53
    model_path = pathlib.Path(__file__).parent.joinpath("yolov5l-face.onnx")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    detector = YOLO5Face(str(model_path), 0.53, 0.5, "cuda")

    for img_path in img_paths:
        assert os.path.exists(img_path), f"Image not found: {img_path}"

        # Load image
        img: np.ndarray = cv2.imread(str(img_path))
        assert img is not None

        # Detect face
        pred = detector.detect_one(img)

        # Make label
        if pred is not None:
            bbox, _, landmarks = detector.parse_prediction(pred)
            bbox = xyxy2xywhn(bbox, img.shape[1], img.shape[0])
            landmarks[..., 0::2] /= img.shape[1]
            landmarks[..., 1::2] /= img.shape[0]
            bbox = np.round(bbox, 6)
            landmarks = np.round(landmarks, 6)

            label = []
            for bbox_one, landmarks_one in zip(bbox, landmarks):
                label.append(
                    f"0 {' '.join([f'{i:.6f}' for i in bbox_one])} {' '.join([f'{i:.6f}' for i in landmarks_one])}\n"
                )

            label_path = root.joinpath("labels", img_path.parts[-2], img_path.parts[-1].replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.writelines(label)
        else:
            no_faces_dir_train_dir = root.joinpath("images", "no_faces", "train")
            no_faces_dir_val_dir = root.joinpath("images", "no_faces", "val")
            no_faces_dir_train_dir.mkdir(parents=True, exist_ok=True)
            no_faces_dir_val_dir.mkdir(parents=True, exist_ok=True)
            if img_path.parts[-2] == "train":
                shutil.move(img_path, no_faces_dir_train_dir)
            elif img_path.parts[-2] == "val":
                shutil.move(img_path, no_faces_dir_val_dir)
            else:
                raise ValueError(f"Wrong img_path: {img_path}")

        # Save detection results
        if pred is not None:
            bbox, conf, landmarks = detector.parse_prediction(pred)
            draw_prediction(img, bbox.astype(np.int32), conf.tolist(), landmarks.astype(np.int32), 1)
        cv2.imwrite(str(root.joinpath("images", "results").joinpath(img_path.name)), img)

        progress_bar.update.remote(1)


def main():
    num_gpus = 8
    print(f"{num_gpus=}")
    ray.init(num_gpus=num_gpus)

    # Initialize
    shutil.rmtree(root.joinpath("labels", "train"), ignore_errors=True)
    shutil.rmtree(root.joinpath("labels", "val"), ignore_errors=True)
    shutil.rmtree(root.joinpath("images", "results"), ignore_errors=True)
    root.joinpath("labels", "train").mkdir()
    root.joinpath("labels", "val").mkdir()
    root.joinpath("images", "results").mkdir()

    img_paths = sorted([i for i in root.joinpath("images").rglob("**/*.jpg")])
    img_paths_split = [i.tolist() for i in np.array_split(img_paths, num_gpus)]

    progress_bar = remote_tqdm.remote(total=len(img_paths))
    ray.get([worker.remote(img_paths_split[i], i, progress_bar) for i in range(num_gpus)])
    ray.shutdown()


if __name__ == '__main__':
    main()
