import os
import sys
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import copy
import cv2
import numpy as np
import math
import time
import traceback

import utility 
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient

logger = get_logger()


class TextRecognizer(object):
    def __init__(self):
        full_dict_path  = '/home/lakiet/study/Projects/Text-detection-pipeline/scripts/ppocr/utils/en_dict.txt'
        self.dictionary = (open(full_dict_path, 'r').read().split('\n'))
        self.dictionary_map = {x: i for i, x in enumerate(self.dictionary)}
        self.max_length = 50

        self.url = '192.168.1.10:8001'
        self.model_name = 'ocr'
        self.triton_client = grpcclient.InferenceServerClient(url = self.url, verbose = False)
    
    def decode(self, encoded_text):
        text = ''
        for item in encoded_text:
            if item == -1:
                continue
            char = self.dictionary[int(item)]
            text += char
        return text

    def __call__(self, img):
        # Create client for triton server for preprocess and inference
        inputs = []
        outputs = []
        
        ori_im_shape = list(img.shape)

        inputs.append(grpcclient.InferInput('images', ori_im_shape, 'UINT8'))

        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))
        outputs.append(grpcclient.InferRequestedOutput('text_rec_output'))
        outputs.append(grpcclient.InferRequestedOutput('text_rec_output_score'))
        
        inputs[0].set_data_from_numpy(img)

        results = self.triton_client.infer(model_name = self.model_name,
                                           inputs = inputs,
                                           outputs = outputs)

        det_boxes = results.as_numpy('text_det_output')
        texts = results.as_numpy('text_rec_output')
        scores = results.as_numpy('text_rec_output_score')

        for i, (encoded_text, score) in enumerate(zip(texts, scores)):
            plain_text = self.decode(encoded_text)
            print(plain_text, score)
            print(det_boxes[0][i])
        
        return det_boxes, texts, scores



