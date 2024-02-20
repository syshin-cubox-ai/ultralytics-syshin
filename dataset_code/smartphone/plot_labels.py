import pathlib

import cv2
import numpy as np
import tqdm

import ultralytics.utils.ops

root = pathlib.Path("D:/data/smartphone").resolve(strict=True)


def main():
    image_paths = sorted([i for i in root.joinpath("images").rglob("**/*.jpg")])
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])

    output_path = root.joinpath("plots")
    output_path.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_path.joinpath("val").mkdir(parents=True, exist_ok=True)

    image_path: pathlib.Path
    label_path: pathlib.Path
    for idx, (image_path, label_path) in tqdm.tqdm(enumerate(zip(image_paths, label_paths, strict=True)),
                                                   desc="Plot",
                                                   total=len(image_paths)):
        # Read files
        image = cv2.imread(str(image_path))
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        # Convert label
        label = []
        for label_one in raw_label:
            label_one = label_one.split(" ")[1:]
            label_one = [float(i) for i in label_one]
            label.append(label_one)
        label = ultralytics.utils.ops.xywhn2xyxy(np.array(label), image.shape[1], image.shape[0]).astype(np.int32)

        # Plot
        for label_one in label:
            x1, y1, x2, y2 = label_one
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save plotted images
        cv2.imwrite(str(output_path.joinpath(*image_path.parts[-2:])), image)


if __name__ == "__main__":
    main()
