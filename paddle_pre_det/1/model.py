import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.load(args['model_config'])
        output_config = pb_utils.get_ouput_config_by_name(self.model_config, 'output')
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, 'images').as_numpy()
            imgs = np.transpose(imgs, (0,2,3,1))
            results = []
            for img in imgs:
                img = img.astype(np.uint8).copy()
                img = img / 255.0
                

        return responses

    def finalize(self):
        print('Cleaning up...')