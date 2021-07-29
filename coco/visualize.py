from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage.io as io
import os

# Option
dataDir='/data/sung/dataset/low-illumination-dataset/coco_exdark'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


# initialize COCO api for instance annotations
coco=COCO(annFile)

cat = coco.loadCats(coco.getCatIds())
print(cat)

#%%

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['car']);
print(catIds)

imgIds = coco.getImgIds(catIds=catIds);
img = coco.loadImgs(imgIds[0])[0]

# Load Image
plt.figure()
I = io.imread(os.path.join(dataDir, dataType, img['file_name']))
plt.axis('off')
plt.imshow(I)

# load and display instance annotations
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns, draw_bbox=True)
plt.show()

#%%


