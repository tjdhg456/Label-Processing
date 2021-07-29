from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
import skimage.io as io

dataDir='/data_2/sung/dataset/furniture/connection/connector_seg/'
dataType='val2017'
annFile='{}/annotations/clip_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

cat = coco.loadCats(coco.getCatIds())
print(cat)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['group'])
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)
img = coco.loadImgs(imgIds[0])[0]
print(img)

I = io.imread(os.path.join(dataDir, 'val2017', img['file_name']))

plt.figure()
plt.axis('off')
plt.imshow(I)
plt.savefig('aaa.png')
plt.show()

# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns, draw_bbox=False)
plt.savefig('bbb.png')
