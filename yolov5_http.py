import torch
import sys
import numpy as np
import cv2
from copy import deepcopy
import tritonclient.http as httpclient
from onnx.yolov5_onnx.common import letterbox, non_max_suppression

model_name = "yolov5_onnx"
url = '192.168.1.10:8000'

try:
    triton_client = httpclient.InferenceServerClient(url=url,verbose=False)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

# Health metadata
if not triton_client.is_server_live():
    print("FAILED : is_server_live")
    sys.exit(1)

if not triton_client.is_server_ready():
    print("FAILED : is_server_ready")
    sys.exit(1)

if not triton_client.is_model_ready(model_name):
    print("FAILED : is_model_ready")
    sys.exit(1)

# Metadata
metadata = triton_client.get_server_metadata()
if not (metadata["name"] == "triton"):
    print("FAILED : get_server_metadata")
    sys.exit(1)
print(metadata)

# Configuration
config = triton_client.get_model_config(model_name)
if not (config['name']== model_name):
    print("FAILED: get_model_config")
    sys.exit(1)

#Input images
image = cv2.imread('test/kaka.jpg') 
im = letterbox(image,new_shape=(640,640), stride=32, auto=False)[0] 
im = im.transpose((2, 0, 1))[::-1]  
im = im / 255
imgs = np.expand_dims(im, axis=0).astype(np.float32)

inputs = []
inputs.append(httpclient.InferInput("images", [1, 3, 640, 640], "FP32"))
inputs[0].set_data_from_numpy(imgs)

#Output image
outputs = []
outputs.append(httpclient.InferRequestedOutput("output0"))

results = triton_client.infer(
    model_name=model_name,
    inputs=inputs,
    outputs=outputs
)

output0_data = results.as_numpy("output0")
output0_copy = deepcopy(output0_data)
output0_copy = [torch.from_numpy(output0_copy)]
output0_copy = non_max_suppression(output0_copy)
print(output0_copy)
