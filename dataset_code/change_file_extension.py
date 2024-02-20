import pathlib

import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)


def main():
    image_paths = sorted(list(root.joinpath("images").rglob("**/*.jpeg")))
    for image_path in tqdm.tqdm(image_paths):
        image_path.rename(image_path.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
