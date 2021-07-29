import json
import glob
from tqdm import tqdm
import numpy as np
import os
from shapely.geometry.polygon import LinearRing, LineString
from shapely.geometry import MultiPolygon, Polygon, Point, GeometryCollection
from shapely.ops import cascaded_union
import cv2
import xml.dom.minidom
import numpy as np
import shapely.geometry as sg
from pycococreatortools.pycococreatortools import create_annotation_info
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pg
from PIL import Image, ImageDraw


PRIMITIVES = ['stick', 'ring', 'plate', 'disk+cylinder', 'disk', 'cylinder', 'cuboids', 'cuboid+cylinder', 'cuboid', 'bended-cylinder', 'bended-cuboid+cylinder', 'bended-cuboid+cuboid', 'bended-cuboid', 'unknown']

def check_collision(bbox, poly, tol):
    xmin, ymin, xmax, ymax = bbox
    x, y = xmin-tol, ymin-tol
    w = xmax - xmin + 2*tol
    h = ymax - ymin + 2*tol

    bbox = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)])    
    if bbox.intersects(poly):
        return True
    else:
        return False

def mask_for_polygons(polygons, im_size, tol=None, color=255, img=None):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    # img_mask = np.zeros(im_size)
    
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(int)
    
    # new_polygons = []
    # for poly in polygons: 
    #     if isinstance(poly, MultiPolygon):
    #         poly = list(poly)
    #         for pi in poly:
    #             new_polygons.append(pi)
    #     else:
    #         new_polygons.append(poly)
    # polygons = new_polygons

    # if isinstance(polygons, MultiPolygon):
    #     print("ohdfdfadsfasdfsd")

    exteriors = [int_coords(poly.buffer(tol, resolution=16, join_style=2, mitre_limit=1).exterior.coords) for poly in polygons if poly.buffer(tol, resolution=16, join_style=2, mitre_limit=1).exterior is not None]
    interiors = [poly for poly in polygons if not isinstance(poly, GeometryCollection)]


    # interiors = [int_coords(pi.coords) for poly in polygons if not isinstance(poly, GeometryCollection)
    #               for pi in poly.interiors ]

    img = Image.new('L', list(reversed(im_size)), 0)

    for i in range(len(exteriors)):
        exterior = [tuple(x) for x in exteriors[i]]
        ImageDraw.Draw(img).polygon(exterior, outline=1, fill=color)
        for pi in interiors[i].interiors:
            interior = int_coords(pi.coords)
            interior = [tuple(x) for x in interior]
            ImageDraw.Draw(img).polygon(interior, outline=1, fill=0)

    return img

def partition(poly_a, poly_b):
    """
    Splits polygons A and B into their differences and intersection.
    """
    if not poly_a.intersects(poly_b):
        return poly_a, poly_b, None

    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)

    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)

    only_a = poly_a.difference(poly_b)
    only_b = poly_b.difference(poly_a)

    inter  = poly_a.intersection(poly_b)
    return only_a, only_b, inter

def overwrite(poly_a, poly_b):
    """
    overwrite poly_a to poly_b
    """
    if not poly_a.intersects(poly_b):
        return poly_a, poly_b
    else:
        only_a, only_b, inter = partition(poly_a, poly_b)
        poly_a = cascaded_union([only_a, inter])
        poly_b = only_b

    return poly_a, poly_b
    
def get_polygon(annotation):

    exterior = annotation['points']['exterior']
    interiors = annotation['points']['interior']
    exterior = [tuple(p) for p in exterior]
    exterior.append(tuple(exterior[0]))

    if not interiors:
        polygon = Polygon(exterior)
        return polygon
    else: 
        new_interiors = []
        for interior in interiors:
            interior = [tuple(p) for p in interior]
            interior.append(tuple(interior[0]))
            new_interiors.append(interior)
        polygon = Polygon(exterior, new_interiors)
        if not polygon.is_valid:
            return polygon.buffer(0)
        else:
            return polygon

def create_image_info(ann_sly, image_id, ann_path):

    if ann_path[-5:] == '.json':
        file_name = ann_path.split('/')[-1][:-5]
        if file_name[-4:] == '.jpg':
            file_name = file_name[:-4]
            file_name = file_name + '.png'


    image = {
        "id": image_id,
        "width": ann_sly['size']['width'],
        "height":ann_sly['size']['height'],
        "file_name": file_name,
        "license": 1
    }

    return image

def create_annotation_from_polys(coco_output, polys, image_id, category_id, annotation_id, ann_sly, tol):

    im_size = (ann_sly['size']['height'], ann_sly['size']['width'])
    mask = np.array(mask_for_polygons(polys, im_size, tol=tol))
    category_info = {'id': category_id, 'is_crowd': 0}
    annotation = create_annotation_info(annotation_id, image_id, category_info, mask, tolerance = tol)

    if annotation is not None:
        coco_output["annotations"].append(annotation)
    # for poly in polys:
    #     if isinstance(poly, MultiPolygon):
    #         for pi in poly.geoms:
    #             if isinstance(pi, GeometryCollection):
    #                 continue
    #             annotation = create_polygon_annotation(pi, image_id, category_id, annotation_id, ann_sly, tolerance)
    #             if annotation is not None:
    #                 coco_output["annotations"].append(annotation)
    #     else:
    #         if isinstance(poly, GeometryCollection):
    #             continue
    #         annotation = create_polygon_annotation(poly, image_id, category_id, annotation_id, ann_sly, tolerance)
    #         if annotation is not None:
    #             coco_output["annotations"].append(annotation)
    return coco_output

def get_bboxs_from_VOC(file_name):

    bboxs = []
    xml_path = '../VOC/step/' + file_name + '.xml'
    try: 
        dom = xml.dom.minidom.parse(xml_path)
    except FileNotFoundError:
        return None

    size = dom.getElementsByTagName("size")[0]           
    root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    
    for i, _object in enumerate(objects):
        bndbox = root.getElementsByTagName('bndbox')[i]
        name = _object.getElementsByTagName('name')[0]
        name = str(name.childNodes[0].data)
        if name == 'step':
            xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
            bboxs.append([xmin, ymin, xmax, ymax])
    return bboxs

def create_bbox_to_COCO_annotation(image_id, category_id, annotation_id, tol, bbox):
        
    segmentation = []
    xmin, ymin, xmax, ymax = bbox
    x, y = xmin-tol, ymin-tol
    w = xmax - xmin + 2*tol
    h = ymax - ymin + 2*tol

    bbox = (x, y, w, h)
    area = w * h

    segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]

    annotation = {
        'iscrowd': 0,
        'image_id': image_id,
        'segmentation': segmentation,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation


def create_polyline_annotation(annotation, image_id, category_id, annotation_id):

    segmentation = []

    line = [tuple(p) for p in annotation['points']['exterior']]
    annotation = {
        'iscrowd': 0,
        'image_id': image_id,
        'segmentation': None,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': None,
        'area': None,
        'line': line
    }
    return annotation

def create_bbox_annotation(annotation, image_id, category_id, annotation_id, tol, type='bbox'):
    
    if type == 'bbox':
        if len(annotation['points']['exterior']) > 2:
            xmin, xmax, ymin, ymax = 50000, 0, 50000, 0
            for x, y in annotation['points']['exterior']:
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
            left_top, right_bottom = [xmin, ymin], [xmax, ymax]
        else:
            left_top, right_bottom = annotation['points']['exterior']
        x, y = left_top
        right, bottom = right_bottom
        w = right - x
        h = bottom - y

    elif type == 'point':
        rad = 5
        x_ori, y_ori = annotation['points']['exterior'][0]
        x, y = x_ori-rad, y_ori-rad
        w, h = 2*rad, 2*rad

    else:
        raise('Select correct type!')

    bbox = (x-tol, y-tol, w+2*tol, h+2*tol) # x, y, w, h
    area = w * h

    segmentation = [[x-tol, y-tol, x+w+tol, y-tol, x+w+tol, y+h+tol, x-tol, y+h+tol]]

    annotation = {
        'iscrowd': 0,
        'image_id': image_id,
        'segmentation': segmentation,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation

def create_polygon_annotation(annotation, image_id, category_id, annotation_id, ann_sly, tol):
    if isinstance(annotation, Polygon):
        poly = annotation
        exterior = poly.buffer(tol, resolution=16, join_style=2, mitre_limit=1).exterior.coords
        interiors = [pi.coords for pi in poly.interiors if not isinstance(poly, GeometryCollection)]

    else:
        exterior = annotation['points']['exterior']
        interiors = annotation['points']['interior']
        exterior = [tuple(p) for p in exterior]

        exterior = LinearRing(exterior)
        exterior = exterior.buffer(tol, resolution=16, join_style=2, mitre_limit=1).exterior.coords

    if len(interiors) == 0:
        is_crowd = 0

        target_region = exterior
        segmentation = []
        for (x, y) in target_region:
            segmentation.append(x)
            segmentation.append(y)
        segmentation = [segmentation]
        exterior = np.asarray(exterior)
        min_x, min_y = np.min(exterior, 0)
        max_x, max_y = np.max(exterior, 0)
        w = max_x - min_x
        h = max_y - min_y
        
        bbox = (int(min_x), int(min_y), int(w), int(h))
        area = int(w*h)

        annotation = {
            'iscrowd': is_crowd,
            'image_id': image_id,
            'segmentation': segmentation,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }

        return annotation

    else: 
        is_crowd = 1
        holes = []
        for interior in interiors:
            interior = [tuple(p) for p in interior]
            # !TODO: support tolerance for holes

            # interior = LinearRing(interior)
            # interior = interior.parallel_offset(-5, resolution=16, join_style=2, mitre_limit=1)
            # if (interior.type != 'LineString') or interior.is_empty == True:
            #     print("This hole is too small to draw an off set. Ignore this hole ...")
            #     continue
            holes.append(interior)

        target_region = Polygon(exterior, holes=holes)
        img_size = (ann_sly['size']['height'], ann_sly['size']['width'])
        mask = mask_for_polygons([target_region], img_size)
        category_info = {'id': category_id, 'is_crowd': is_crowd}

        annotation = create_annotation_info(annotation_id, image_id, category_info, mask, tolerance = tol)
        return annotation

def intermediates(p1, p2, nb_points=10):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    output = []
    output += p1
    for i in range(1, nb_points+1):
        output += [p1[0] + i * x_spacing, p1[1] +  i * y_spacing]
    output += p2
    return [output]

def create_line_annotation(annotation, image_id, category_id, annotation_id, tol):
    ann_points = annotation['points']['exterior']
    x_ori, y_ori = ann_points[0], ann_points[1]
    bbox = [int(np.min([x_ori[0], y_ori[0]])), int(np.min([x_ori[1], y_ori[1]])), int(np.abs(x_ori[0]-y_ori[0])), int(np.abs(x_ori[1]-y_ori[1]))]

    if bbox[2] < 20:
        bbox[0] = bbox[0] - int((10 - bbox[2]) / 2)
        bbox[2] = 20
    if bbox[3] < 20:
        bbox[1] = bbox[1] - int((10 - bbox[3]) / 2)
        bbox[3] = 20

    area = bbox[2] * bbox[3]

    segmentation = intermediates(x_ori, y_ori, nb_points=10)
    annotation = {
        'iscrowd': 0,
        'image_id': image_id,
        'segmentation': segmentation,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation

