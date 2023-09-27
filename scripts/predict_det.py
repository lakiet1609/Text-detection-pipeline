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
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
import tritonclient.grpc as grpcclient
import json
logger = get_logger()


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx

        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = 'fast'
        postprocess_params["box_types"] = 'quad'

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(args, 'det', logger)

        self.model_name = "paddle_infer_text_det"
        self.url = '192.168.1.10:8001'

        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        img_from_triton = deepcopy(img)
        
        inputs = []
        outputs = []
        img_from_triton = np.expand_dims(img_from_triton, axis=0)
        img_from_triton = np.transpose(img_from_triton, (0, 3, 1, 2))
        img_tri_shape = list(img_from_triton.shape)
        
        inputs.append(grpcclient.InferInput('images', img_tri_shape, "UINT8"))
        
        outputs.append(grpcclient.InferRequestedOutput('output'))
        outputs.append(grpcclient.InferRequestedOutput('shape_list'))

        inputs[0].set_data_from_numpy(img_from_triton)
        
        results = self.triton_client.infer(model_name="paddle_pre_det",
                                           inputs=inputs,
                                           outputs=outputs)
        
        img_from_triton_ = results.as_numpy('output')[0]
        shape_list_from_triton = results.as_numpy('shape_list')[0]
        print(img_from_triton_.shape)
        print(shape_list_from_triton.shape)


        # Infer
        img_shape = list(img_from_triton_.shape)
        
        inputs = []
        inputs.append(grpcclient.InferInput("x", img_shape, "FP32"))
        inputs[0].set_data_from_numpy(img_from_triton_)
    
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("sigmoid_0.tmp_0"))
        results =self.triton_client.infer(model_name=self.model_name,
                                          inputs=inputs,
                                          outputs=outputs)
        
        output = results.as_numpy("sigmoid_0.tmp_0")
        preds = {}
        preds['maps'] = output
        
        #Postprocess
        post_result = self.postprocess_op(preds, shape_list_from_triton)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, img.shape)

        return dt_boxes


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
 

