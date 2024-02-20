import os
import pathlib

import cv2
import numpy as np
import tqdm

from dataset_code.coco_face.yolo5face import resize_preserving_aspect_ratio

root = pathlib.Path("D:/data/bodyhands").resolve(strict=True)
limit = 2000

img_paths = sorted([i for i in root.joinpath("images").rglob("**/*.jpg")])
for img_path in tqdm.tqdm(img_paths):
    assert os.path.exists(img_path), f"Image not found: {img_path}"

    # Load image
    img: np.ndarray = cv2.imread(str(img_path))
    assert img is not None

    # Limit resolution
    if max(img.shape) > limit:
        img, _ = resize_preserving_aspect_ratio(img, limit)
        cv2.imwrite(str(img_path), img)
