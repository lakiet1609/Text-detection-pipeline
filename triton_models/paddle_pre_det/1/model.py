import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "output")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()  
            imgs = np.transpose(imgs,(0,2,3,1))
            results = []
            for img in imgs:
                img = img.astype(np.uint8).copy()
                img = img / 255.0
                img = img / 0.227
                img = img - 0.45
                img = np.transpose(img, (2,0,1))
                results.append(img)

            results = np.array(results)
            results = np.ascontiguousarray(results, dtype=self.output_dtype)
            out_tensor_0 = pb_utils.Tensor("output", results)

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')