import pathlib

import tqdm

src_root = pathlib.Path("D:/data/camera").resolve(strict=True)
dst_root = pathlib.Path("D:/data/phone").resolve()
move_class_id = 0


def main():
    dst_root.joinpath("images", "train").mkdir(parents=True, exist_ok=True)
    dst_root.joinpath("images", "val").mkdir(parents=True, exist_ok=True)
    dst_root.joinpath("labels", "train").mkdir(parents=True, exist_ok=True)
    dst_root.joinpath("labels", "val").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(src_root.joinpath("images").rglob("**/*.jpg")))
    label_paths = sorted(list(src_root.joinpath("labels").rglob("**/*.txt")))

    image_path: pathlib.Path
    label_path: pathlib.Path
    for image_path, label_path in tqdm.tqdm(zip(image_paths, label_paths, strict=True), total=len(image_paths)):
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        flag = False
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id == move_class_id:
                flag = True

        if flag:
            image_path.rename(dst_root.joinpath(*image_path.parts[-3:]))
            label_path.rename(dst_root.joinpath(*label_path.parts[-3:]))


if __name__ == "__main__":
    main()
