import json
import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
import xml.dom.minidom
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from PIL import Image
from utility import create_sub_masks, create_annotation_info

def create_group_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    img_id = 0
    ann_id = 0

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        ann_img = Image.open(ann_path)

        # Image info
        img_dict = {}
        img_dict['file_name'] = ann_path.split('/')[-1]
        img_dict['height'] = ann_img.size[1]
        img_dict['width'] = ann_img.size[0]
        img_dict['id'] = img_id

        coco_output["images"].append(img_dict)

        # Sub masks
        ann_img = np.array(ann_img)
        ann_list = sorted(np.unique(ann_img))[1:]
        for key in ann_list:
            sub_mask = np.where(ann_img == key, 1, 0)
            category_info = {'id': 0, 'is_crowd': 0}
            ann_imp = create_annotation_info(ann_id, img_id, category_info, sub_mask, tolerance=3)
            coco_output["annotations"].append(ann_imp)
            ann_id += 1

        img_id += 1

    # write final annotation file
    base = os.path.dirname(save_file)
    os.makedirs(base, exist_ok=True)
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')


