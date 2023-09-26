docker run --gpus all --shm-size=1G --rm  \
 -p8000:8000 -p8001:8001 -p8002:8002 \
 -v $PWD/triton_models:/models \
 nvcr.io/nvidia/tritonserver:22.05-py3_new tritonserver --model-repository=/models

