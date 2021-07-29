import json
import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
import xml.dom.minidom
import numpy as np
from pycococreatortools.pycococreatortools import create_annotation_info
from matplotlib import pyplot as plt
import utility
from collections import defaultdict
import json

def merge_ann_list(ann_list, id):
    default = ann_list[0]

    # Change Segmentation, id, bbox, area
    bbox_list = []
    seg_list = []
    for ann in ann_list:
        x, y, w, h = ann['bbox']
        bbox_list.append([x, y, x+w, y+h])
        seg_list.append(ann['segmentation'][0])

    default['segmentation'] = seg_list
    bbox_list = np.array(bbox_list)
    x1,y1,x2,y2 = np.min(bbox_list[:,0]), np.min(bbox_list[:,1]), np.max(bbox_list[:,2]), np.max(bbox_list[:,3])
    area = (x2-x1) * (y2-y1)
    default['bbox'] = (int(x1), int(y1), int(x2-x1), int(y2-y1))
    default['area'] = int(area)
    default['id'] = int(id)
    default['category_id'] = 0
    return default


def create_bond_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    image_id = 0
    annotation_id = 0

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        with open(ann_path) as json_file:
            ann_sly = json.load(json_file)

        # create image info
        image_info = utility.create_image_info(ann_sly, image_id, ann_path)
        coco_output["images"].append(image_info)

        # Load annotated label in a single annotation file
        ann_dict = defaultdict(list)
        for annotation in ann_sly['objects']:
            ann_class = annotation['classTitle']
            if annotation['tags'] == []:
                ann_tag = []
                ann_old_tag = []
            else:
                ann_old_tag = [an['name'] for an in annotation['tags']]
                ann_tag = [an['name'] for an in annotation['tags'] if an['name'] in ['1','2','3','4','5','6','7','8','9','10']]

            if len(ann_tag) == 1:
                bonding = True

                ann_tag = ann_tag[0]
                category_id = 0

                if (ann_class == 'POINT') or (ann_class == 'CONNECTOR'):
                    type='point'
                else:
                    type='bbox'

            elif ann_class == 'pair':
                bonding = False
                category_id = 3
                type='bbox'

            elif ann_class == 'CONNECTOR':
                bonding = False
                category_id = 2
                type = 'point'

            elif ann_class == 'POINT':
                bonding = False
                category_id = 1
                type = 'point'

            else:
                bonding = False
                category_id = None
                type=None

            # write object for hole and connector
            if category_id is not None:
                if (ann_class == 'POINT') and ('out' not in ann_old_tag):
                    # Category_id
                    category_id = 1

                elif (ann_class == 'CONNECTOR') and ('UNSEEN' not in ann_old_tag) and ('out' not in ann_old_tag):
                    # Category_id
                    category_id = 2
                    b1, b2 = annotation['points']['exterior']
                    annotation['points']['exterior'] = [[int((b1[0]+b2[0])/2), int((b1[1]+b2[1])/2)]]

                elif (ann_class == 'pair'):
                    category_id = 3
                else:
                    continue

                # create annotation
                new_annotation = utility.create_bbox_annotation(annotation, image_id, category_id, annotation_id,
                                                                tolerance, type=type)
                if new_annotation is not None:
                    coco_output["annotations"].append(new_annotation)
                    annotation_id += 1

            # Write object for bonding
            if (category_id is not None) and (bonding):
                new_annotation = utility.create_bbox_annotation(annotation, image_id, category_id, annotation_id,
                                                                tolerance, type=type)
                if new_annotation is not None:
                    ann_dict[ann_tag].append(new_annotation)
            else:
                continue

        # Merge annotation dictionary
        if len(ann_dict.keys()) > 0:
            for key in ann_dict.keys():
                ann_list = ann_dict[key]
                ann_imp = merge_ann_list(ann_list, annotation_id)
                coco_output["annotations"].append(ann_imp)
                annotation_id += 1
        else:
            continue

        image_id += 1

    # write final annotation file
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')

def create_hole_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    image_id = 1
    annotation_id = 1

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        with open(ann_path) as json_file:
            ann_sly = json.load(json_file)

        # create image info
        image_info = utility.create_image_info(ann_sly, image_id, ann_path)
        coco_output["images"].append(image_info)

        # Load annotated label in a single annotation file
        for annotation in ann_sly['objects']:
            ann_class = annotation['classTitle']
            if annotation['tags'] == []:
                ann_tag = []
            else:
                ann_tag = [an['name'] for an in annotation['tags']]


            # Select hole and connector label
            if (ann_class == 'POINT') and ('out' not in ann_tag) and ('UNSEEN' not in ann_tag):
                # Category_id
                category_id = 0
            elif (ann_class == 'CONNECTOR') and ('UNSEEN' not in ann_tag) and ('out' not in ann_tag):
                # Category_id
                category_id = 1
            else:
                category_id = None

            # Write object!
            if category_id is not None:
                # create annotation
                type = 'point' if category_id == 0 else 'bbox'
                annotation = utility.create_bbox_annotation(annotation, image_id, category_id, annotation_id, tolerance, type=type)
                if annotation is not None:
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
            else:
                continue

        image_id += 1

    # write final annotation file
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')





def create_text_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    image_id = 1
    annotation_id = 1

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        with open(ann_path) as json_file:
            ann_sly = json.load(json_file)

        # create image info
        image_info = utility.create_image_info(ann_sly, image_id, ann_path)
        coco_output["images"].append(image_info)

        # Load annotated label in a single annotation file
        for annotation in ann_sly['objects']:
            ann_class = annotation['classTitle']

            # Select hole and connector label
            if (ann_class == 'text'):
                # Category_id
                category_id = 0
            else:
                category_id = None

            # Write object!
            if category_id is not None:
                # create annotation
                type = 'bbox'
                annotation = utility.create_bbox_annotation(annotation, image_id, category_id, annotation_id, tolerance, type=type)
                if annotation is not None:
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
            else:
                continue

        image_id += 1

    # write final annotation file
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')

def create_line_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    image_id = 1
    annotation_id = 1

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        with open(ann_path) as json_file:
            ann_sly = json.load(json_file)

        # create image info
        image_info = utility.create_image_info(ann_sly, image_id, ann_path)
        coco_output["images"].append(image_info)

        # Load annotated label in a single annotation file
        for annotation in ann_sly['objects']:
            ann_class = annotation['classTitle']

            # Select hole and connector label
            if (ann_class == 'line_new'):
                # Category_id
                category_id = 0

            else:
                category_id = None

            # Write object!
            if category_id is not None:
                tol = 2
                annotation = utility.create_line_annotation(annotation, image_id, category_id, annotation_id, tol)

                if annotation is not None:
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
            else:
                continue

        image_id += 1

    # write final annotation file
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')


def create_challenge_connection_annotation(ann_file, save_file, dataset_info, licenses, categories, tolerance):
    coco_output = {
        "info": dataset_info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ann_paths = glob.glob(ann_file)
    ann_paths.sort()

    image_id = 1
    annotation_id = 1

    # go through all image file
    for _, ann_path in enumerate(tqdm(ann_paths)):
        # Load single annotation file
        with open(ann_path) as json_file:
            ann_sly = json.load(json_file)

        # create image info
        file_name = os.path.basename(ann_path).rstrip('.json')
        image_info = {
            "id": image_id,
            "width": ann_sly['size']['width'],
            "height":ann_sly['size']['height'],
            "file_name": file_name,
            "license": 1
        }

        coco_output["images"].append(image_info)

        # Load annotated label in a single annotation file
        for annotation in ann_sly['objects']:
            ann_class = annotation['classTitle']

            # Select hole and connector label
            if (ann_class == 'line_new'):
                # Category_id
                category_id = 0
            elif (ann_class == 'connector'):
                # Category_id
                category_id = 1
            elif (ann_class == 'region'):
                # Category_id
                category_id = 2
            else:
                category_id = None

            # Write object!
            if category_id is not None:
                tol = 2
                if category_id == 0:
                    annotation = utility.create_line_annotation(annotation, image_id, category_id, annotation_id, tol)
                elif category_id in [1,2]:
                    annotation = utility.create_bbox_annotation(annotation, image_id, category_id, annotation_id, tol, type='bbox')

                if annotation is not None:
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
            else:
                continue


        # Load Group
        with open('/home/sung/dataset/furniture/detection/assembly_challenge/anno_group/%s' %file_name.replace('.png','.json'), 'r') as f:
            group = json.load(f)

        for ix in range(group['group']['num_instance']):
            category_id = 3
            bbox = group['group']['bbox'][str(ix+1)]
            area = int(bbox[2] * bbox[3])
            annotation = {
                'iscrowd': 0,
                'image_id': image_id,
                'segmentation': [],
                'category_id': category_id,
                'id': annotation_id,
                'bbox': bbox,
                'area': area
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # write final annotation file
    with open(save_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('saved')