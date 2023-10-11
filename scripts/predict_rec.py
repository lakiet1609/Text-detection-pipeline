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
    def __init__(self, args):
        full_dict_path  = '/home/lakiet/study/Projects/Text-detection-pipeline/scripts/ppocr/utils/en_dict.txt'
        self.dictionary = (open(full_dict_path, 'r').read().split('\n'))
        self.dictionary_map = {x: i for i, x in enumerate(self.dictionary)}
        self.max_length = 50

        self.url = '192.168.1.10:8001'
        self.model_name = 'paddle_text_rec'
        self.triton_client = grpcclient.InferenceServerClient(url = self.url, verbose = False)
    
    def decode(self, encoded_text):
        text = ''
        for item in encoded_text:
            if item == -1:
                continue
            char = self.dictionary[int(item)]
            text += char
        return text

    def __call__(self, img, dt_boxes):
        # Create client for triton server for preprocess and inference
        inputs = []
        outputs = []
        
        ori_im_shape = list(img.shape)
        dt_boxes_shape = list(dt_boxes.shape)
        
        inputs.append(grpcclient.InferInput('pre_images', ori_im_shape, 'UINT8'))
        inputs.append(grpcclient.InferInput('pre_dt_boxes', dt_boxes_shape, 'FP32'))

        outputs.append(grpcclient.InferRequestedOutput('post_rec_output'))
        outputs.append(grpcclient.InferRequestedOutput('post_rec_output_score'))
        
        inputs[0].set_data_from_numpy(img)
        inputs[1].set_data_from_numpy(dt_boxes)

        results = self.triton_client.infer(model_name = self.model_name,
                                           inputs = inputs,
                                           outputs = outputs)

        texts = results.as_numpy('post_rec_output')
        scores = results.as_numpy('post_rec_output_score')

        # # Create client for triton server for post-process
        # inputs = []
        # outputs = []
        
        # infer_rec_result_shape = list(infer_rec_result.shape)
        
        # inputs.append(grpcclient.InferInput('post_rec_input', infer_rec_result_shape, 'FP32'))

        # outputs.append(grpcclient.InferRequestedOutput('post_rec_output'))
        # outputs.append(grpcclient.InferRequestedOutput('post_rec_output_score'))
        
        # inputs[0].set_data_from_numpy(infer_rec_result)

        # results = self.triton_client.infer(model_name = 'paddle_post_rec',
        #                                    inputs = inputs,
        #                                    outputs = outputs)

        # texts = results.as_numpy('post_rec_output')
        # scores = results.as_numpy('post_rec_output_score')

        for encoded_text, score in zip(texts, scores):
            plain_text = self.decode(encoded_text)
            print(plain_text, score)
        
        return texts, scores



