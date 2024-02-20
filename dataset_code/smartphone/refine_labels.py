import pathlib

import numpy as np
import ultralytics.utils.ops

root = pathlib.Path("D:/data/smartphone").resolve(strict=True)


def main():
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])

    for label_path in label_paths:
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        # Convert label
        label = []
        for label_one in raw_label:
            label_one = label_one.split(" ")[1:]
            label_one = [float(i) for i in label_one]
            if len(label_one) != 4:
                label_one = np.array(label_one)[:8]
                label_one = np.array(
                    [np.min(label_one[::2]), np.min(label_one[1::2]), np.max(label_one[::2]),
                     np.max(label_one[1::2])])
                label_one = ultralytics.utils.ops.xyxy2xywh(label_one)
            label.append(label_one)

        label_str = []
        for label_one in label:
            label_str.append(f"0 {' '.join([f'{i:.6f}' for i in label_one])}\n")
        with open(label_path, "w") as f:
            f.writelines(label_str)


if __name__ == "__main__":
    main()
