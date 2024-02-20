import pathlib
import shutil

import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)


def main():
    image_paths = sorted(list(root.joinpath("images").rglob("**/*.*")))

    new_images = root.joinpath("new_images")
    new_images.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_images.joinpath("val").mkdir(parents=True, exist_ok=True)
    new_labels = root.joinpath("new_labels")
    new_labels.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_labels.joinpath("val").mkdir(parents=True, exist_ok=True)

    for image_path in tqdm.tqdm(image_paths):
        label_path = root.joinpath("labels", image_path.parent.name, f"{image_path.stem}.txt")
        image_path.rename(new_images.joinpath(*image_path.parts[-2:]))
        label_path.rename(new_labels.joinpath(*label_path.parts[-2:]))

    shutil.rmtree(root.joinpath("images"))
    shutil.rmtree(root.joinpath("labels"))
    shutil.move(new_images, root.joinpath("images"))
    shutil.move(new_labels, root.joinpath("labels"))


if __name__ == "__main__":
    main()
