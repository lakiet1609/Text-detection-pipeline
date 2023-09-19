docker run --name triton --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/lakiet/study/Projects/Text-detection-pipeline/triton_models:/models nvcr.io/nvidia/tritonserver:22.05-py3 tritonserver --model-repository=/models