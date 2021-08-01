# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from pathlib import Path
from pycocotools.coco import COCO
import requests
import os
from tqdm import tqdm
import json
import numpy as np

def annotate_coco_subset(ann_file, save_file, class_list):
    coco = COCO(ann_file)
    catId = coco.getCatIds(catNms=class_list)
    cat = coco.loadCats(ids=catId)

    category = []
    conversion = dict()

    # Update Class
    for ix, cls in enumerate(cat):
        conversion[str(cls['id'])] = ix+1   
        cls['id'] = ix+1
        category.append(cls)

    # Load Annotation
    annId = coco.getAnnIds(catIds=catId)
    ann = coco.loadAnns(ids=annId)

    for ix in range(len(ann)):
        ann[ix]['category_id'] = conversion[str(ann[ix]['category_id'])]

    # Load Image list
    img_id_list = []
    for c in catId:
        img_id = coco.getImgIds(catIds=c)
        img_id_list += img_id

    img_id_list = list(set(img_id_list))
    img = coco.loadImgs(ids=img_id_list)

    # Load JSON
    with open(ann_file, 'r') as f:
        file = json.load(f)

    # Update
    file['images'] = img
    file['annotations'] = ann
    file['categories'] = category

    # Save JSON
    with open(save_file, 'w') as f:
        json.dump(file, f)


def download_partial_coco(category, base_folder, save_folder):
    # instantiate COCO specifying the annotations json path
    coco = COCO(os.path.join(base_folder, 'annotations', 'instances_train2017.json'))

    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=category)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    os.makedirs(os.path.join(base_folder, save_folder), exist_ok=True)
    print('Total Length : %d' %len(images))

    for im in tqdm(images):
        img_data = requests.get(im['coco_url']).content
        with open(os.path.join(base_folder, save_folder, im['file_name']), 'wb') as handler:
            handler.write(img_data)


if __name__=='__main__':
    base = '/data/sung/dataset/coco'
    class_path = os.path.join(base, 'coco_labels.txt')
    with open(class_path, 'r') as f:
        class_name = f.readlines()
    
    class_name = [class_.strip().split(',')[-1] for class_ in class_name]
    class_name_9 = np.random.choice(class_name, int(len(class_name) * 0.9), replace=False)
    class_name_1 = np.array(list(set(class_name) - set(class_name_9)))
    

    # Train
    annotate_coco_subset(os.path.join(base, 'annotations', 'instances_train2017.json'),
                        os.path.join(base, 'annotations', 'partial', 'part_0.9_train2017.json'),
                        class_name_9)

    annotate_coco_subset(os.path.join(base, 'annotations', 'instances_train2017.json'),
                        os.path.join(base, 'annotations', 'partial', 'part_0.1_train2017.json'),
                        class_name_1)
    
    # Val
    annotate_coco_subset(os.path.join(base, 'annotations', 'instances_train2017.json'),
                        os.path.join(base, 'annotations', 'partial', 'part_0.9_train2017.json'),
                        class_name_9)

    annotate_coco_subset(os.path.join(base, 'annotations', 'instances_train2017.json'),
                        os.path.join(base, 'annotations', 'partial', 'part_0.1_train2017.json'),
                        class_name_1)
    