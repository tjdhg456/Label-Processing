import os
import shutil
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage.io as io
import os
from collections import OrderedDict
import json
from coco_subset import annotate_coco_subset

def merge(json1, json2, save_name):
    # Merge Category
    # Json1
    coco1 = COCO(json1)
    cat1Id = coco1.getCatIds()
    cat1 = coco1.loadCats(cat1Id)

    img1Id = coco1.getImgIds()
    img1 = coco1.loadImgs(img1Id)

    ann1Id = coco1.getAnnIds()
    ann1 = coco1.loadAnns(ann1Id)

    # Json2
    coco2 = COCO(json2)
    cat2Id = coco2.getCatIds()
    cat2 = coco2.loadCats(cat2Id)

    img2Id = coco2.getImgIds()
    img2 = coco2.loadImgs(img2Id)

    ann2Id = coco2.getAnnIds()
    ann2 = coco2.loadAnns(ann2Id)

    # Print Category
    print(cat1)
    print(cat2)

    # Intersection of Image
    img1_name = set([img['file_name'] for img in img1])
    img2_name = set([img['file_name'] for img in img2])
    img_name = list(img1_name.intersection(img2_name))
    img_dict = OrderedDict([(img, ix) for ix, img in enumerate(sorted(img_name))])

    # Convert Dictionary for image ID
    img1_dict = OrderedDict()
    img2_dict = OrderedDict()

    img_list = []
    for img in img1:
        if img['file_name'] in img_name:
            # Make common img_list
            old_id = img['id']
            img['id'] = img_dict[img['file_name']]
            img_list.append(img)

            # Make converting dictionary
            img1_dict[old_id] = img['id']
        else:
            continue

    for img in img2:
        if img['file_name'] in img_name:
            # Make converting dictionary
            old_id = img['id']
            new_id = img_dict[img['file_name']]
            img2_dict[old_id] = new_id
        else:
            continue

    # Convert Dictionary for category ID
    ix = 0
    cat1_dict = OrderedDict()
    cat2_dict = OrderedDict()

    cat_list = []
    for cat in cat1:
        cat1_dict[cat['id']] = ix
        cat['id'] = ix
        ix += 1
        cat_list.append(cat)

    for cat in cat2:
        cat2_dict[cat['id']] = ix
        cat['id'] = ix
        ix += 1
        cat_list.append(cat)

    # Converting annotations
    ann_list = []
    ix = 0
    for ann in ann1:
        if ann['image_id'] in img1_dict.keys():
            ann['image_id'] = img1_dict[ann['image_id']]
            ann['category_id'] = cat1_dict[ann['category_id']]
            ann['id'] = ix
            ix += 1
            ann_list.append(ann)
        else:
            continue

    for ann in ann2:
        if ann['image_id'] in img2_dict.keys():
            ann['image_id'] = img2_dict[ann['image_id']]
            ann['category_id'] = cat2_dict[ann['category_id']]
            ann['id'] = ix
            ix += 1
            ann_list.append(ann)
        else:
            continue

    # Load JSON
    with open(json1, 'r') as f:
        file = json.load(f)

    # Update
    file['images'] = img_list
    file['annotations'] = ann_list
    file['categories'] = cat_list

    # Save JSON
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, 'w') as f:
        json.dump(file, f)

def assembly_clip(class_list, ann_file):
    type = ann_file.rstrip('.json').split('_')[-1]

    ## Basic Options
    old_base = '/'.join(ann_file.split('/')[:-2])
    save_base = os.path.join('/'.join(ann_file.split('/')[:-3]), ann_file.split('/')[-3] + '_clip')

    old_base = os.path.join(old_base, '%s' %type)
    save_base = os.path.join(save_base, '%s' %type)
    os.makedirs(save_base,exist_ok=True)

    ## Load ann_file
    coco = COCO(ann_file)

    ## Dictionary for converting original imageId into clipped imageId
    assemblyCatId = coco.getCatIds(catNms=['assembly'])
    assemblyAnnId = coco.getAnnIds(catIds = assemblyCatId)
    assemblyAnn = coco.loadAnns(ids=assemblyAnnId)
    assemblyDict = OrderedDict([])
    for ix, ann in enumerate(assemblyAnn):
        if ann['image_id'] in assemblyDict.keys():
            assemblyDict[ann['image_id']].append((ann['bbox'], ix))
        else:
            assemblyDict[ann['image_id']] = [(ann['bbox'], ix)]

    ## Convert Ann
    img_list = []
    for cls in class_list:
        cls = ['assembly', cls]
        catId = coco.getCatIds(catNms=cls)
        imgId = coco.getImgIds(catIds=catId)
        img_list += imgId

    # For Unique Image List
    targetImgId = list(set(img_list))
    targetCatId = coco.getCatIds(catNms=class_list)
    targetAnnId = coco.getAnnIds(imgIds=targetImgId, catIds=targetCatId)
    ann = coco.loadAnns(ids=targetAnnId)

    ann_list = []
    ix = 0
    for a in ann:
        for bboxDict, idDict in assemblyDict[a['image_id']]:
            if inbox(bboxDict, a['bbox']) == True:
                a['bbox'] = convertCoordinate(bboxDict, a['bbox'], type='bbox')
                a['segmentation'] = convertCoordinate(bboxDict, a['segmentation'], type='segmentation')
                a['image_id'] = idDict
                a['id'] = ix
                ix += 1

                ann_list.append(a)
                break
            else:
                continue

    ## Clip and Convert Image
    img_list = []
    for t_id in targetImgId:
        loadImg = coco.loadImgs(t_id)[0]
        bbox_imp = assemblyDict[t_id]
        for bbox, img_id in bbox_imp:
            img = {}
            # Convert
            old_name = loadImg['file_name']
            new_name = loadImg['file_name'].rstrip('.png') + '_%d.png' %img_id
            img['file_name'] = new_name
            img['width'] = bbox[2]
            img['height'] = bbox[3]
            img['id'] = img_id

            img_list.append(img)

            # Crop
            crop(os.path.join(old_base, old_name), os.path.join(save_base,new_name), bbox)

    ## Save Annotations file
    with open(ann_file, 'r') as f:
        file = json.load(f)

    # Update
    file['images'] = img_list
    file['annotations'] = ann_list

    # Save JSON
    save_name = os.path.join(save_base, '../annotations', 'clip_%s.json'%type)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, 'w') as f:
        json.dump(file, f)

    annotate_coco_subset(save_name, save_name, class_list)


# Check whether the center of bbox_small is located inside of the bbox_large
def inbox(bbox_large, bbox_small):

    def center(bbox):
        x,y,w,h = bbox
        center = [x + int(w/2), y + int(h/2)]
        return center

    x1,y1,w1,h1 = bbox_large
    center2 = center(bbox_small)

    if (x1 <= center2[0] <= (x1+w1)) and (y1 <= center2[1] <= (y1+h1)):
        return True
    else:
        return False

# Convert the bbox coordinations based on criterion
def convertCoordinate(criterion, old, type):
    x,y,w,h = criterion

    if type == 'bbox':
        x2,y2,w2,h2 = old
        new = [x2-x, y2-y, w2, h2]
    elif type == 'segmentation':
        new = []
        for seg in old:
            seg_list = []
            for ix, s in enumerate(seg):
                if ix % 2 == 0:
                    new_s = s-x
                else:
                    new_s = s-y
                seg_list.append(new_s)
            new.append(seg_list)
    return new

# clip image using the given bounding box
import cv2
def crop(filename, save_name, bbox):
    x,y,w,h = bbox
    image = plt.imread(filename)
    clip_image = np.array(image[y:y+h , x:x+w, :]) * 255

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    cv2.imwrite(save_name, clip_image)


if __name__=='__main__':
    # Base directory
    base = '/data/sung/dataset/furniture/detection/'

    # Merge - train
    assembly_train = os.path.join(base,'assembly_assembly,number/annotations/assembly_train2017.json')
    connector_train = os.path.join(base, 'assembly_connector/annotations/connector_train2017.json')
    merge(assembly_train, connector_train, os.path.join(base, 'assembly_connector/annotations/merge_train2017.json'))

    # Merge - validation
    assembly_val = os.path.join(base,'assembly_assembly,number/annotations/assembly_val2017.json')
    connector_val = os.path.join(base, 'assembly_connector/annotations/connector_val2017.json')
    merge(assembly_val, connector_val, os.path.join(base, 'assembly_connector/annotations/merge_val2017.json'))

    # Clip - both train and validation
    class_list = ['hole']
    assembly_clip(class_list, os.path.join(base, 'assembly_hole/annotations/merge_train2017.json'))
    assembly_clip(class_list, os.path.join(base, 'assembly_hole/annotations/merge_val2017.json'))
