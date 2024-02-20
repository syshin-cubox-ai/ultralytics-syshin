import pathlib

import cv2
import numpy as np
import tqdm

import ultralytics.utils.ops

root = pathlib.Path("D:/data/camera").resolve(strict=True)


def put_border_text(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float,
                    color=(255, 255, 255), border_color=(0, 0, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, border_color, 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)


def plot(
        img: np.ndarray,
        pred: np.ndarray,
        line_width=3,
        font_size=0.65,
        show=False,
        save=False,
        filename: str = None,
) -> np.ndarray:
    palette = np.array([(0, 255, 0), (56, 56, 255), (31, 112, 255),
                        (0, 255, 255), (187, 212, 0), (199, 55, 255)], np.uint8)
    plotted_img = img.copy()

    if pred.size > 0:
        for pred_one in pred:
            cls, box = np.split(pred_one, [1])
            cls, box = cls.astype(np.int32).item(), box.astype(np.int32)

            # box
            x1, y1, x2, y2 = box
            cv2.rectangle(plotted_img, (x1, y1), (x2, y2), palette[cls].tolist(), line_width)

            # class_id, confidence
            put_border_text(plotted_img, f"{cls}", (x1, y1 - 3),
                            font_size, palette[cls].tolist())

    if show:
        cv2.imshow("0", plotted_img)
    if save:
        cv2.imwrite(filename, plotted_img)
    return plotted_img


def main():
    image_paths = sorted(list(root.joinpath("images").rglob("**/*.jpg")))
    label_paths = sorted(list(root.joinpath("labels").rglob("**/*.txt")))

    output_path = root.joinpath("plots")
    output_path.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_path.joinpath("val").mkdir(parents=True, exist_ok=True)

    image_path: pathlib.Path
    label_path: pathlib.Path
    for idx, (image_path, label_path) in tqdm.tqdm(enumerate(zip(image_paths, label_paths, strict=True)),
                                                   total=len(image_paths)):
        # Read files
        image = cv2.imread(str(image_path))
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        # Convert label
        if len(raw_label) > 0:
            label = []
            for label_one in raw_label:
                label_one = label_one.split(" ")
                label_one = [float(i) for i in label_one]
                label.append(label_one[:5])
            label = np.array(label)
            label[:, 1:] = ultralytics.utils.ops.xywhn2xyxy(label[:, 1:], image.shape[1], image.shape[0])

            # Plot
            plotted_image = plot(image, label)
        else:
            plotted_image = image.copy()

        # Save plotted images
        cv2.imwrite(str(output_path.joinpath(*image_path.parts[-2:])), plotted_image)


if __name__ == "__main__":
    main()
