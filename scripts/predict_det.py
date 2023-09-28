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

import utility as utility
from ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient
import json
logger = get_logger()

class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(args, 'det', logger)
        self.model_name = "paddle_infer_text_det"
        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)

    def __call__(self, img):
        #Text Preprocess
        img_from_triton = deepcopy(img)
        inputs = []
        outputs = []
        img_from_triton = np.expand_dims(img_from_triton, axis=0)
        img_from_triton = np.transpose(img_from_triton, (0, 3, 1, 2))
        img_tri_shape = list(img_from_triton.shape)
        
        inputs.append(grpcclient.InferInput('images', img_tri_shape, "UINT8"))
        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))
        inputs[0].set_data_from_numpy(img_from_triton)
        results = self.triton_client.infer(model_name="paddle_text_det",
                                           inputs=inputs,
                                           outputs=outputs)
        
        text_det_output = results.as_numpy('text_det_output')
        return text_det_output

if __name__ == "__main__":
    args = utility.parse_args()
    
    draw_img_save = "./inference_results"
    
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    
    # image_file_list = get_image_file_list(args.image_dir)
    
    text_detector = TextDetector(args)
    
    img_path = args.image_dir
    img = cv2.imread(img_path)
    dt_boxes = text_detector(img)
    
    src_im = utility.draw_text_det_res(dt_boxes, img_path)
    
    cv2.imwrite(os.path.join(draw_img_save, 'result.jpg'), src_im)
 

