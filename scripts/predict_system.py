import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import logging
from PIL import Image
import utility
import predict_rec
import predict_det

from ppocr.utils.logging import get_logger

logger = get_logger()

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)
        self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = args.drop_score
        self.args = args
        self.crop_image_res_index = 0

    def __call__(self, img):
        ori_im = img.copy()
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        
        dt_boxes = self.text_detector(img)
        rec_res = self.text_recognizer(img, dt_boxes)
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            _, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res

def main(args):
    text_sys = TextSystem(args)
    img = cv2.imread(args.image_dir)
    imgs = [img]
    for index, img in enumerate(imgs):
        dt_boxes, rec_res = text_sys(img)


if __name__ == "__main__":
    args = utility.parse_args()
    main(args)
