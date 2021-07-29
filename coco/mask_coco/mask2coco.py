'''
convert annotation format from supervisely & VOC to COCO format
'''

import os
import datetime
from convert import *
import sys
import argparse
import glob

INFO = {
"description": "Assembly Instruction Dataset",
"url": "https://github.com/gist-ailab/mmdetection",
"version": "0.2.0",
"year": 2020,
"contributor": "changhyeon chun, sungho shin",
"date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
}
]

GROUP_CATEGORIES = [
{
    'id': 0,
    'name': 'group',
    'supercategory': 'group'
}
]



if __name__ == "__main__":
    tolerance = 3

    # Generate COCO formated annotation
    group_base = '/home/sung/dataset/assembly_group/'

    create_group_annotation(os.path.join(group_base,'mask_png', 'val', 'mask', '*.png'),
                           os.path.join(group_base, 'annotations', 'group_val2017.json'),
                           INFO, LICENSES, GROUP_CATEGORIES, tolerance)

    create_group_annotation(os.path.join(group_base,'mask_png', 'test', 'mask', '*.png'),
                           os.path.join(group_base, 'annotations', 'group_test2017.json'),
                           INFO, LICENSES, GROUP_CATEGORIES, tolerance)

    create_group_annotation(os.path.join(group_base,'mask_png', 'train', 'mask', '*.png'),
                           os.path.join(group_base, 'annotations', 'group_train2017.json'),
                           INFO, LICENSES, GROUP_CATEGORIES, tolerance)






