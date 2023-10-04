# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
# import paddle

import utility 
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient

logger = get_logger()


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(args, 'rec', logger)
        self.url = '192.168.1.10:8001'
        # self.model_name = 'paddle_text_det'
        self.triton_client = grpcclient.InferenceServerClient(url = self.url, verbose = False)

    def __call__(self, img, dt_boxes):
        # Create client for triton server for preprocess
        inputs = []
        outputs = []
        
        ori_im_shape = list(img.shape)
        dt_boxes_shape = list(dt_boxes.shape)
        
        inputs.append(grpcclient.InferInput('pre_images', ori_im_shape, 'UINT8'))
        inputs.append(grpcclient.InferInput('pre_dt_boxes', dt_boxes_shape, 'FP32'))

        outputs.append(grpcclient.InferRequestedOutput('pre_rec_output'))
        
        inputs[0].set_data_from_numpy(img)
        inputs[1].set_data_from_numpy(dt_boxes)

        results = self.triton_client.infer(model_name = 'paddle_pre_rec',
                                           inputs = inputs,
                                           outputs = outputs)

        pre_rec_outputs = results.as_numpy('pre_rec_output')
        
        # Infer 
        # Create client for triton server
        inputs = []
        outputs = []
        pre_rec_outputs_shape = list(pre_rec_outputs.shape)
        
        inputs.append(grpcclient.InferInput('x', pre_rec_outputs_shape, 'FP32'))
        outputs.append(grpcclient.InferRequestedOutput('softmax_2.tmp_0'))
        
        inputs[0].set_data_from_numpy(pre_rec_outputs)

        results = self.triton_client.infer(model_name = 'paddle_infer_text_rec',
                                           inputs = inputs,
                                           outputs = outputs)

        infer_text_rec_outputs = results.as_numpy('softmax_2.tmp_0')
        
        #Post-process
        rec_result = self.postprocess_op(infer_text_rec_outputs)

        return rec_result



