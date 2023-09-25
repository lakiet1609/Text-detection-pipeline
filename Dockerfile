FROM nvcr.io/nvidia/tritonserver:22.05-py3
RUN pip install opencv-python
RUN apt-get update && apt-get install -y libgl1

