import triton_python_backend_utils as pb_utils
import cv2
import os
import json
import numpy as np
from ppocr.postprocess import build_post_process
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        
        output_config_0 = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output_config_0["data_type"])

        output_config_1 = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output_score")
        self.output1_dtype = pb_utils.triton_string_to_numpy(output_config_1["data_type"])

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dict_path = "en_dict.txt"
        full_dict_path  = os.path.join(current_dir, dict_path)
        
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": full_dict_path,
            "use_space_char": True
        }
        
        self.postprocess_op = build_post_process(postprocess_params)
        self.dictionary = (open(full_dict_path, 'r').read().split('\n'))
        self.dictionary_map = {x: i for i, x in enumerate(self.dictionary)}
        self.max_length = 50
        print('Initialized...')
    
    def encode(self, text):
        encoded_text = [-1]*self.max_length
        for i, char in enumerate(text):
            encoded_text[i] = self.dictionary_map[char]
            if i == self.max_length-1:
                break
        encoded_text = np.array(encoded_text)
        return encoded_text

    def execute(self, requests):
        responses = []
        for request in requests:
            rec_result = pb_utils.get_input_tensor_by_name(request, "post_rec_input").as_numpy()
            rec_result = self.postprocess_op(rec_result)
            texts = [result[0] for result in rec_result]
            scores = [result[1] for result in rec_result]

            text_result = []
            score_result = []
            for text, score in zip(texts, scores):
                encoded_text = self.encode(text)
                text_result.append(encoded_text)
                score_result.append(score)


            text_res = np.ascontiguousarray(text_result, dtype=self.output0_dtype)
            out_tensor_0 = pb_utils.Tensor("post_rec_output", text_res.astype(self.output0_dtype))
            
            score_res = np.ascontiguousarray(score_result, dtype=self.output1_dtype)
            out_tensor_1 = pb_utils.Tensor("post_rec_output_score", score_res.astype(self.output1_dtype))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')