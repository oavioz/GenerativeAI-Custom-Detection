import cv2
import numpy as np
import math
import argparse
from PIL import Image
from yolo import YOLOv8_face

class FaceDetector:
    def __init__(self, modelpath, confThreshold=0.45, nmsThreshold=0.5) -> None:
        self.model = YOLOv8_face(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)

    def face_detect(self, imgpath):
        img = cv2.imread(imgpath)

        boxes, scores, classids, kpts = self.model.detect(img)

        pad = 10
        crops = []
        for bbox in boxes:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x<0 or y<0 or w<=0 or h<=0 or x+w>img.shape[1] or y+h>img.shape[0]:
                print("problem")
                continue
            
            new_x = x - pad
            if new_x<0:
                new_x = 0
            pad_x = x-new_x
            new_y = y - pad
            if new_y<0:
                new_y = 0
            pad_y = y-new_y
            new_w = w+pad_x+pad
            new_h = h+pad_y+pad
            if new_x+new_w>img.shape[1]:
                new_w = img.shape[1]-new_x
            if new_y+new_h>img.shape[0]:
                new_h = img.shape[0]-new_y
            
            crop = img[new_y:new_y + new_h, new_x:new_x + new_w]
            pil_image = Image.fromarray(crop)

            crops.append(pil_image)
        
        return crops