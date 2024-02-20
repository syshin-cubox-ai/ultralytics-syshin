import pathlib
import random

import tqdm

root = pathlib.Path("D:/data/passport-seg")
num_val = 100

all_images = list(root.joinpath("images", "train").glob("*.jpg"))

random.shuffle(all_images)
train_images = all_images[num_val:]
val_images = all_images[:num_val]
val_stems = [i.stem for i in val_images]

for val_stem in tqdm.tqdm(val_stems):
    src_image_path = root.joinpath("images", "train", f"{val_stem}.jpg")
    src_label_path = root.joinpath("labels", "train", f"{val_stem}.txt")
    dst_image_path = root.joinpath("images", "val", f"{val_stem}.jpg")
    dst_label_path = root.joinpath("labels", "val", f"{val_stem}.txt")

    src_image_path.rename(dst_image_path)
    src_label_path.rename(dst_label_path)
