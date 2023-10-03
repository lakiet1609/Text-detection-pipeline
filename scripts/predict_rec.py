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
from utility import get_rotate_crop_image
import tritonclient.grpc as grpcclient

logger = get_logger()


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
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
        self.model_name = 'paddle_text_det'
        self.triton_client = grpcclient.InferenceServerClient(url = self.url, verbose = False)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def sorted_boxes(self, dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, 0, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def __call__(self, img, dt_boxes):
        
        # # Pre-process for text recoginition
        # if dt_boxes is None:
        #     return None

        # dt_boxes = self.sorted_boxes(dt_boxes)
        # dt_boxes = dt_boxes[0]

        # img_list = []
        # for bno in range(len(dt_boxes)):
        #     tmp_box = copy.deepcopy(dt_boxes[bno])
        #     img_crop = get_rotate_crop_image(ori_im, tmp_box)
        #     img_list.append(img_crop)
        
        # Create client for triton server
        inputs = []
        outputs = []
        
        ori_im_shape = list(img.shape)
        dt_boxes_shape = list(dt_boxes.shape)
        print(dt_boxes_shape)
        
        inputs.append(grpcclient.InferInput('pre_images', ori_im_shape, 'UINT8'))
        inputs.append(grpcclient.InferInput('pre_dt_boxes', dt_boxes_shape, 'FP32'))

        outputs.append(grpcclient.InferRequestedOutput('pre_rec_output'))
        
        inputs[0].set_data_from_numpy(img)
        inputs[1].set_data_from_numpy(dt_boxes)

        results = self.triton_client.infer(model_name = 'paddle_pre_rec',
                                           inputs = inputs,
                                           outputs = outputs)

        pre_rec_outputs = results.as_numpy('pre_rec_output')
        
        # width_list = []
        # for img in img_list:
        #     width_list.append(img.shape[1] / float(img.shape[0]))

        # max_wh_ratio = max(width_list)
        # imgC, imgH, imgW = self.rec_image_shape[:3]
        # setting_max_wh_ratio = imgW / imgH
        # max_wh_ratio = max(max_wh_ratio, setting_max_wh_ratio)

        # norm_img_batch = []
        # for img in img_list:
        #     norm_img = self.resize_norm_img(img,max_wh_ratio)
        #     norm_img = norm_img[np.newaxis, :]
        #     norm_img_batch.append(norm_img)

        # norm_img_batch = np.concatenate(norm_img_batch)
        # norm_img_batch = norm_img_batch.copy()
        # print(type(norm_img_batch))

        # Infer 
        # Create client for triton server
        inputs = []
        outputs = []
        pre_rec_outputs_shape = list(pre_rec_outputs.shape)
        
        inputs.append(grpcclient.InferInput('x', pre_rec_outputs_shape, 'FP32'))
        outputs.append(grpcclient.InferRequestedOutput('softmax_2.tmp_0'))
        
        inputs[0].set_data_from_numpy(pre_rec_outputs_shape)

        results = self.triton_client.infer(model_name = 'paddle_infer_text_rec',
                                           inputs = inputs,
                                           outputs = outputs)

        infer_text_rec_outputs = results.as_numpy('softmax_2.tmp_0')
        
        #Post-process
        rec_result = self.postprocess_op(infer_text_rec_outputs)

        return rec_result



