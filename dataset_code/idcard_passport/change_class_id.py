import pathlib

import tqdm

root = pathlib.Path("D:/data/passport-seg")
dst_class_id = 1

label_paths = list(root.joinpath("labels").rglob("**/*.txt"))
for label_path in tqdm.tqdm(label_paths):
    label = label_path.read_text("utf-8")
    label_path.write_text(f"{dst_class_id}{label[1:]}", "utf-8")
