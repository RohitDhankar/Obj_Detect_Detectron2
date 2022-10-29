
""" 
conda activate env2_det2
COCO DATA -- MS COCO -->>  https://cocodataset.org/#detection-eval
COCO DATA Download -- SOURCE -->> https://www.neuralception.com/cocodatasetapi/
"""

import torch, detectron2
#!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
"""
torch:  1.10 ; cuda:  1.10.1
detectron2: 0.6
"""

#Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

## First steps with COCO Data 

#!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
coco_test_image = cv2.imread("./input.jpg")
print("---type(coco_test_image",type(coco_test_image)) ## ---type(coco_test_image <class 'numpy.ndarray'>


## INIT ANNO 
import json
path_validate = "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/coco_train_2017/annotations/instances_val2017.json"
path_train = "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/coco_train_2017/annotations/instances_train2017.json"

# f_train_annos = open(path_train)
# anno_train = json.load(f_train_annos)
# print("---ANNO--Keys---",anno_train.keys())
# print("---ANNO--Keys---",anno_train["images"][1])
# print("---ANNO--Keys----annotations-",anno_train["annotations"][1])

from pycocotools.coco import COCO

coco_obj=COCO(path_validate)
#print(type(coco_obj)) ##<class 'pycocotools.coco.COCO'>

# Get list of category_ids, here [2] for bicycle
category_ids = coco_obj.getCatIds(['bicycle'])

# Get list of image_ids which contain bicycles
image_ids = coco_obj.getImgIds(catIds=[2])
print("--10 IMAGE ID's----",image_ids[0:10])

ls_image_ids_bikes = image_ids[0:10]
print("-ls_image_ids_bikes---",type(ls_image_ids_bikes))


"""
TRAIN SET
loading annotations into memory...
Done (t=9.13s)
creating index...
index created!
<class 'pycocotools.coco.COCO'>
[196610, 344067, 155652, 417797, 294918, 57353, 516105, 253965, 229391, 57361]
"""

"""
VALIDATION SET 
[184324, 546823, 343561, 169996, 277005, 549390, 384527, 8211, 93717, 468505]

"""

# Get all bicycle annotations for TRAIN SET --image 000000196610.jpg
# Get all bicycle annotations for VAL SET --  image 000000093717.jpg

#image_id = 343561 #93717 #169996

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

for iter_image in range(len(ls_image_ids_bikes)):
    #
    print("---iter_image-",ls_image_ids_bikes[iter_image])

    #annotation_ids = coco_obj.getAnnIds(imgIds=196610, catIds=[2])
    annotation_ids = coco_obj.getAnnIds(imgIds=ls_image_ids_bikes[iter_image], catIds=[2])
    print(len(annotation_ids))

    ls_anno_ids = coco_obj.loadAnns(annotation_ids)
    print("---ls_anno_ids----\n",ls_anno_ids)

    # for anno in ls_anno_ids:
    #     print(anno['bbox'])

    images_path_validate = "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/val2017/"
    image_name = str(ls_image_ids_bikes[iter_image]).zfill(12)+".jpg" # Image names are 12 characters long
    image = Image.open(images_path_validate+image_name)
    fig, ax = plt.subplots()

    # ax.imshow(image)
    # plt.show()

    # Draw boxes and add label to each box
    for anno in ls_anno_ids:
        box = anno['bbox']
        bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=3, edgecolor="blue", facecolor="none")
        ax.add_patch(bb)

    ax.imshow(image)
    plt.show()

