# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# https://paperswithcode.github.io/torchbench/
RUN pip install --upgrade pip
RUN pip install \
    torchbench \
    torchvision \
    timm \
    Pillow \
    matplotlib \
    numpy \
    pennylane \
    tensorflow \
    torch \
    pyyaml \
    hyperopt \
    tensorboardX
RUN mkdir /data
RUN mkdir /ckpt
RUN mkdir /src
WORKDIR /src
COPY traineval.py src/traineval.py
COPY model.py src/model.py
CMD ["python", "src/traineval.py"]