import glob
import json
import os

img_dir = os.path.join('data', 'idcard_trainset')
annotation_path = os.path.join('data', 'idcard_trainset.json')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Check the number of images and number of annotations
num_images = len(glob.glob(os.path.join(img_dir, '*.jpg')))
assert num_images == len(coco['images']) == len(coco['annotations'])
print(f'전체 개수: {num_images}')

# Map image id and image file name.
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Check annotations
multiple_polygon = []
invalid_num_polygon_points = []
multiple_category = []
invalid_num_bbox_points = []
for anno in coco['annotations']:
    if not len(anno['segmentation']) == 1:
        multiple_polygon.append(image_id_filename[anno['image_id']])
    if not len(anno['segmentation'][0]) == 8:
        invalid_num_polygon_points.append(image_id_filename[anno['image_id']])
    if not anno['category_id'] == 1:
        multiple_category.append(image_id_filename[anno['image_id']])
    if not len(anno['bbox']) == 4:
        invalid_num_bbox_points.append(image_id_filename[anno['image_id']])

if len(multiple_polygon) or len(invalid_num_polygon_points) or len(multiple_category) or len(invalid_num_bbox_points):
    print(f'polygon이 여러 개: {multiple_polygon}')
    print(f'polygon의 점 개수가 잘못됨: {invalid_num_polygon_points}')
    print(f'category 개수가 잘못됨: {multiple_category}')
    print(f'bbox의 점 개수가 잘못됨: {invalid_num_bbox_points}')
else:
    print('annotation 이상 없음')
