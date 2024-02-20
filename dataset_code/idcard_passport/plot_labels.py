import pathlib

import cv2
import numpy as np
import tqdm

root = pathlib.Path("D:/data/idcard-seg").resolve(strict=True)


def main():
    image_paths = sorted([i for i in root.joinpath("images").rglob("**/*.jpg")])
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])

    output_path = root.joinpath("plots")
    output_path.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_path.joinpath("val").mkdir(parents=True, exist_ok=True)

    image_path: pathlib.Path
    label_path: pathlib.Path
    for image_path, label_path in tqdm.tqdm(zip(image_paths, label_paths, strict=True), desc="Plot",
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
        label = np.array(label)
        label[..., 0::2] = label[..., 0::2] * image.shape[1]
        label[..., 1::2] = label[..., 1::2] * image.shape[0]
        label = label.round().astype(np.int32)

        # Plot polygon area
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, label.reshape(label.shape[0], 4, 2), (0, 0, 255))
        alpha = 0.45
        gamma = 50
        plotted_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, gamma)

        # Save plotted images
        cv2.imwrite(str(output_path.joinpath(*image_path.parts[-2:])), plotted_image)


if __name__ == "__main__":
    main()
