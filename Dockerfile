FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install torchbench
WORKDIR /src
COPY traineval.py src/traineval.py
CMD ["python", "src/traineval.py"]