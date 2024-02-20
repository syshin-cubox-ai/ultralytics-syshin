import glob
import json
import os

import cv2
import tqdm

img_dir = os.path.join('data', 'idcard_testset')
annotation_path = os.path.join('data', 'idcard_testset_original.json')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Check the number of images and number of annotations
img_paths = [os.path.basename(i) for i in glob.glob(os.path.join(img_dir, '*.jpg'))]

refined_coco = {
    'images': [],
    'categories': coco['categories'],
    'annotations': [],
}
for img_path in img_paths:
    filename = os.path.basename(img_path)
    coco_image = [i for i in coco['images'] if i['file_name'] == filename]
    if not len(coco_image) == 1:
        assert len(coco_image) > 1
        print(f"{coco_image[0]['file_name']}")
        coco_image = coco_image[1]
    else:
        coco_image = coco_image[0]
    refined_coco['images'].append(coco_image)
    coco_anno = [i for i in coco['annotations'] if i['image_id'] == coco_image['id']]
    assert len(coco_anno) == 1
    refined_coco['annotations'].append(coco_anno[0])

refined_coco['images'].sort(key=lambda x: x['id'])
refined_coco['annotations'].sort(key=lambda x: x['id'])

with open(annotation_path.replace('_original', ''), 'w', encoding='utf-8') as f:
    json.dump(refined_coco, f)

image_id_filename = {image['id']: image['file_name'] for image in refined_coco['images']}
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)
for anno in tqdm.tqdm(refined_coco['annotations'], 'debug'):
    x1, y1, w, h = anno['bbox']
    x2, y2 = x1 + w, y1 + h

    filename = image_id_filename[anno['image_id']]
    image_path = os.path.join(img_dir, filename)
    img = cv2.imread(image_path)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_dir, filename), img)
