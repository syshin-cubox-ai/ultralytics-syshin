import pathlib
import shutil

# Source Path
widerface_root = pathlib.Path("D:/data/widerface").resolve(strict=True)
coco_face_root = pathlib.Path("D:/data/coco-face").resolve(strict=True)

# Destination Path
face_mix_root = pathlib.Path("D:/data/face-mix").resolve()
shutil.rmtree(face_mix_root, ignore_errors=True)

# Make dataset structure
face_mix_root.mkdir()
face_mix_root.joinpath("images", "train").mkdir(parents=True)
face_mix_root.joinpath("images", "val").mkdir(parents=True)
face_mix_root.joinpath("labels", "train").mkdir(parents=True)
face_mix_root.joinpath("labels", "val").mkdir(parents=True)

# Move widerface data
widerface_src_paths = [i for i in widerface_root.rglob("**/*.*")]
for src_path in widerface_src_paths:
    parts = list(src_path.parts)
    parts[-4] = face_mix_root.name
    dest_path = pathlib.Path(*parts)
    src_path.rename(dest_path)
print("wideface dataset processing completed.")

# Move coco-face data
coco_face_src_paths = [i for i in coco_face_root.joinpath("images", "train").rglob("**/*.*")]
coco_face_src_paths = coco_face_src_paths + [i for i in coco_face_root.joinpath("labels", "train").rglob("**/*.*")]
for src_path in coco_face_src_paths:
    parts = list(src_path.parts)
    parts[-4] = face_mix_root.name
    dest_path = pathlib.Path(*parts)
    src_path.rename(dest_path)
print("coco-face dataset processing completed.")

# Remove useless files
shutil.rmtree(widerface_root)
shutil.rmtree(coco_face_root)
