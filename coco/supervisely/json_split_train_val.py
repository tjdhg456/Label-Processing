import os
from glob import glob
from tqdm import tqdm
import numpy as np
import shutil

# Basic options
type = 'json'
base = '/data_2/sung/dataset/furniture/detection/'
type_folder = 'Annotations_hole'

# Path
base_folder = os.path.join(base, 'assembly_dataset')
save_base = os.path.join(base, 'assembly_connector')

furniture_list = True
furniture_list_folder = os.path.join(base, 'assembly_dataset')

# Unique Image Name
img_list = []
if type == 'json':
    # VOC
    ann_list = glob(os.path.join(base_folder, 'annotations/%s/*.json' %type_folder))

    if furniture_list is None:
        for ann in ann_list:
            # Load annotation file
            img_name = os.path.basename(ann)
            # ann = os.path.dirname(ann)

            # Set connector name
            img_name = '_'.join(img_name.split('_')[:-1])
            img_list.append(img_name)

        # Unique Image Set
        img_unique = list(set(img_list))
        length = len(img_unique)

        # Split index
        train_ix = np.random.choice(range(length), int(length * 0.75), replace=False)
        test_ix = list(set(range(length)) - set(train_ix))

        # Train/Test split
        train_img = np.array(img_unique)[train_ix]
        test_img = np.array(img_unique)[test_ix]

    else:
        train_img = np.load(os.path.join(furniture_list_folder, 'train_list.npy'))
        test_img = np.load(os.path.join(furniture_list_folder, 'val_list.npy'))

    # Load splited images
    os.makedirs(os.path.join(save_base, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(save_base, 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(save_base, 'annotations', 'json', 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_base, 'annotations', 'json', 'val'), exist_ok=True)

    for tr_img in train_img:
        tr_ann_list = tr_img + '_*.json'
        tr_ann_list = os.path.join(base_folder, 'annotations', type_folder, tr_ann_list)
        tr_ann_list = glob(tr_ann_list)

        if tr_ann_list == []:
            print('skip')
            continue
        else:
            for tr_old_ann_name in tr_ann_list:
                ann_base_name = os.path.basename(tr_old_ann_name)
                tr_new_ann_name = os.path.join(save_base, 'annotations', 'json', 'train', ann_base_name)

                img_base_name = ann_base_name.replace('.jpg.json', '.png')
                tr_old_img_name = os.path.join(base_folder, 'manual_png', img_base_name)
                tr_new_img_name = os.path.join(save_base, 'train2017', img_base_name)

                # Copy
                shutil.copy(tr_old_img_name, tr_new_img_name)
                shutil.copy(tr_old_ann_name, tr_new_ann_name)

    for te_img in test_img:
        te_ann_list = te_img + '_*.json'
        te_ann_list = os.path.join(base_folder, 'annotations', type_folder, te_ann_list)
        te_ann_list = glob(te_ann_list)

        if te_ann_list == []:
            print('skip')
            continue
        else:
            for te_old_ann_name in te_ann_list:
                ann_base_name = os.path.basename(te_old_ann_name)
                te_new_ann_name = os.path.join(save_base, 'annotations', 'json', 'val', ann_base_name)

                img_base_name = ann_base_name.replace('.jpg.json', '.png')
                te_old_img_name = os.path.join(base_folder, 'manual_png', img_base_name)
                te_new_img_name = os.path.join(save_base, 'val2017', img_base_name)

                # Copy
                shutil.copy(te_old_img_name, te_new_img_name)
                shutil.copy(te_old_ann_name, te_new_ann_name)

