import triton_python_backend_utils as pb_utils
from ppocr.data import transform, create_operators
import cv2
import json
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config_0 = pb_utils.get_output_config_by_name(self.model_config, "pre_det_output")
        self.output_dtype_0 = pb_utils.triton_string_to_numpy(output_config_0["data_type"])

        output_config_1 = pb_utils.get_output_config_by_name(self.model_config, "pre_det_shape_list")
        self.output_dtype_1 = pb_utils.triton_string_to_numpy(output_config_1["data_type"])
        
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': 'max',
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]

        self.preprocess_op = create_operators(pre_process_list)

        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()  
            imgs = np.transpose(imgs,(0,2,3,1))
            results = []
            results_shape = []
            for img in imgs:
                data = {'image': img}
                data = transform(data, self.preprocess_op)
                img, shape_list = data
                if img is None:
                    return None, 0
                results.append(img)
                results_shape.append(shape_list)

            results = np.array(results)
            results = np.ascontiguousarray(results, dtype=self.output_dtype_0)
            out_tensor_0 = pb_utils.Tensor("pre_det_output", results)
            
            results_shape = np.array(results_shape)
            results_shape = np.ascontiguousarray(results_shape, dtype=self.output_dtype_1)
            out_tensor_1 = pb_utils.Tensor("pre_det_shape_list", results_shape)

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')