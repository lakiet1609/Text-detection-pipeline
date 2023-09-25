import numpy as np
import sys
import cv2
import torch
import copy
import tritonclient.grpc as grpcclient

url = "192.168.1.10:8001"
verbose = False
model_name = "python_backend_for_test"

try:
    triton_client = grpcclient.InferenceServerClient(url=url,verbose=verbose)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

img_path = 'test/kaka.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640,640))
img = np.expand_dims(img, axis=0) #b, w, h, c
img = np.transpose(img, (0, 3, 1, 2))
img_shape = list(img.shape)
# print(img_shape)

#Input
inputs = []
inputs.append(grpcclient.InferInput('images', img_shape, 'UINT8'))
inputs[0].set_data_from_numpy(img)

#Output
outputs = []
outputs.append(grpcclient.InferRequestedOutput('output'))

#Infer
results = triton_client.infer(model_name= model_name,
                              inputs= inputs,
                              outputs= outputs)

images = results.as_numpy('output')

for i, image in enumerate(images):
    cv2.imwrite(f'result_{i}.jpg', image)



