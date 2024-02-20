import pathlib

import tqdm

root = pathlib.Path("/raid/coco-objects365-faceperson").resolve(strict=True)
person_class = 1


def main():
    label_paths = sorted(list(root.joinpath("labels").rglob("**/*.txt")))

    for label_path in tqdm.tqdm(label_paths):
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        label = []
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id == person_class and len(label_one.split()) == 10:
                new_label = label_one.split() + ["0.000000"] * 10
                label.append(f"{' '.join(new_label)}\n")
            else:
                label.append(label_one)

        with open(label_path, "w") as f:
            f.writelines(label)


if __name__ == "__main__":
    main()
