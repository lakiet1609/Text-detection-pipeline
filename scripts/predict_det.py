import os
import sys
from copy import deepcopy

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import sys
import copy

import utility as utility
from ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient
import json
logger = get_logger()

class TextDetector(object):
    def __init__(self):
        self.model_name = "paddle_text_det"
        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)

    def __call__(self, img):
        inputs = []
        outputs = []
        img_tri_shape = list(img.shape)
        
        inputs.append(grpcclient.InferInput('images', img_tri_shape, "UINT8"))
        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))
        inputs[0].set_data_from_numpy(img)
        results = self.triton_client.infer(model_name=self.model_name,
                                           inputs=inputs,
                                           outputs=outputs)
        
        text_det_output = results.as_numpy('text_det_output')
        return text_det_output

if __name__ == "__main__":
    text_detector = TextDetector()

    img1 = cv2.imread("doc/imgs/ger_2.jpg")
    img1 = cv2.resize(img1, (640,640))

    img2 = cv2.imread("doc/imgs/12.jpg")
    img2 = cv2.resize(img2, (640,640))
    imgs = [img1, img2]
    ori_imgs = copy.deepcopy(imgs)
    imgs = np.array(imgs)
    imgs  = np.transpose(imgs, (0, 3, 1, 2))

    dt_boxes= text_detector(imgs)
    print(dt_boxes.shape)
    for i,dt_bbox in enumerate (dt_boxes):
        src_im = utility.draw_text_det_res(dt_boxes[i], ori_imgs[i])
        cv2.imwrite(f"result_{i}.jpg", src_im)
 

