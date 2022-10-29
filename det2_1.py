
""" 
conda activate env2_det2
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
#(env2_det2) dhankar@dhankar-1:~/.../coco_val_images_2017$ wget http://images.cocodataset.org/val2017/

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/unlabeled2017.zip





