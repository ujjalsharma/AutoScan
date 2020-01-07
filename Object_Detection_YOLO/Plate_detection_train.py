import numpy as np
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/plate_detector_yolo.cfg", 
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 0.9,
           "train": True,
           "annotation": "./data/CarImgXML/",
           "dataset": "./data/CarImg/"}

tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()