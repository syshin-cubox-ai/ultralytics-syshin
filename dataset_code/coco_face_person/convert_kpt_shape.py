import pathlib

import tqdm

import dataset_code.coco_face_person.process

# root = pathlib.Path("D:/data/coco-face-person").resolve(strict=True)
root = pathlib.Path("D:/data/coco-face-person-extra").resolve(strict=True)
original_kpt_shape = [5, 2]
new_kpt_shape = [5, 3]


def main():
    label_paths = sorted([i for i in root.joinpath("labels").rglob("**/*.txt")])
    for label_path in tqdm.tqdm(label_paths):
        with open(label_path) as f:
            label = f.readlines()

        new_label = []
        for label_one in label:
            cls, bbox, kpt = label_one.split()[0], label_one.split()[1:5], label_one.split()[5:]
            match int(cls):
                case dataset_code.coco_face_person.process.face_class:
                    add_kpt_value = "2.000000"
                case dataset_code.coco_face_person.process.person_class:
                    add_kpt_value = "0.000000"
                case _:
                    raise ValueError(f"Invalid class: {cls}")

            converted_kpt = kpt.copy()
            for i in range(original_kpt_shape[1], new_kpt_shape[0] * new_kpt_shape[1] + 1, new_kpt_shape[1]):
                converted_kpt.insert(i, add_kpt_value)

            new_label.append(f"{' '.join(label_one.split()[:5] + converted_kpt)}\n")

        with open(label_path, "w") as f:
            f.writelines(new_label)


if __name__ == "__main__":
    main()
