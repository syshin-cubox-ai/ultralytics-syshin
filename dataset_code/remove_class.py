import pathlib

import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)
remove_class_id = 2


def main():
    label_paths = sorted(list(root.joinpath("labels").rglob("**/*.txt")))

    for label_path in tqdm.tqdm(label_paths):
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        label = []
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id != remove_class_id:
                label.append(label_one)

        with open(label_path, "w") as f:
            f.writelines(label)


if __name__ == "__main__":
    main()
