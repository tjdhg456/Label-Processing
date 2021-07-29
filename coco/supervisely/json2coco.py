'''
convert annotation format from supervisely & VOC to COCO format
'''

import os
import datetime
from preprocess_tools import *
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

TEXT_CATEGORIES = [
{
    'id': 0,
    'name': 'text',
    'supercategory': 'text'
}
]

HOLE_CATEGORIES = [
{
    'id': 0,
    'name': 'hole',
    'supercategory': 'hole'
},


{
    'id': 1,
    'name': 'connector',
    'supercategory': 'connector'
}
]


BOND_CATEGORIES = [
{
    'id': 0,
    'name': 'bond',
    'supercategory': 'bond'
},
{
    'id': 1,
    'name': 'hole',
    'supercategory': 'hole'
},
{
    'id': 2,
    'name': 'connector',
    'supercategory': 'connector'
},
{
    'id': 3,
    'name': 'pair',
    'supercategory': 'pair'
}
]


LINE_CATEGORIES = [
{
    'id': 0,
    'name': 'line',
    'supercategory': 'line'
}
]


CHALLENGE_CONNECTION_CATEGORIES = [
{
    'id': 0,
    'name': 'line',
    'supercategory': 'line'
},
{
    'id': 1,
    'name': 'connector',
    'supercategory': 'connector'
},
{
    'id': 2,
    'name': 'region',
    'supercategory': 'region'
},
{
    'id': 3,
    'name': 'group',
    'supercategory': 'group'
}
]



if __name__ == "__main__":
    tolerance = 2

    # Generate COCO formated annotation
    # bond_base = '/home/sung/dataset/furniture/detection/assembly_bond/annotations'
    connector_base = '/data_2/sung/dataset/furniture/detection/assembly_connector/annotations'
    # challenge_connection_base = '/home/sung/dataset/furniture/detection/assembly_challenge_connection/annotations'

    create_hole_annotation(os.path.join(connector_base,'json','train','*.json'),
                           os.path.join(connector_base, 'connector_train2017.json'),
                           INFO, LICENSES, HOLE_CATEGORIES, tolerance)

    create_hole_annotation(os.path.join(connector_base,'json','val','*.json'),
                           os.path.join(connector_base, 'connector_val2017.json'),
                           INFO, LICENSES, HOLE_CATEGORIES, tolerance)


    # line_base = '/home/sung/dataset/furniture/detection/assembly_line/annotations'
    # create_line_annotation(os.path.join(line_base,'json','train','*.json'),
    #                        os.path.join(line_base, 'line_train2017.json'),
    #                        INFO, LICENSES, LINE_CATEGORIES, tolerance)
    #
    # create_line_annotation(os.path.join(line_base,'json','val','*.json'),
    #                        os.path.join(line_base, 'line_val2017.json'),
    #                        INFO, LICENSES, LINE_CATEGORIES, tolerance)


    # text_base = '/home/sung/dataset/furniture/detection/assembly_text/annotations'
    # create_text_annotation(os.path.join(text_base,'json','train','*.json'),
    #                        os.path.join(text_base, 'text_train2017.json'),
    #                        INFO, LICENSES, TEXT_CATEGORIES, tolerance)
    #
    # create_text_annotation(os.path.join(text_base,'json','val','*.json'),
    #                        os.path.join(text_base, 'text_val2017.json'),
    #                        INFO, LICENSES, TEXT_CATEGORIES, tolerance)


    #
    # create_bond_annotation(os.path.join(bond_base,'json','train','*.json'),
    #                        os.path.join(bond_base, 'bond_train2017.json'),
    #                        INFO, LICENSES, BOND_CATEGORIES, tolerance)
    #
    # create_bond_annotation(os.path.join(bond_base,'json','val','*.json'),
    #                        os.path.join(bond_base, 'bond_val2017.json'),
    #                        INFO, LICENSES, BOND_CATEGORIES, tolerance)


    # create_challenge_connection_annotation(os.path.join(challenge_connection_base,'json','train','*.json'),
    #                        os.path.join(challenge_connection_base, 'connection_train2017.json'),
    #                        INFO, LICENSES, CHALLENGE_CONNECTION_CATEGORIES, tolerance)
    #
    # create_challenge_connection_annotation(os.path.join(challenge_connection_base,'json','val','*.json'),
    #                        os.path.join(challenge_connection_base, 'connection_val2017.json'),
    #                        INFO, LICENSES, CHALLENGE_CONNECTION_CATEGORIES, tolerance)





