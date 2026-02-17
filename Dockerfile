FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# NOTE: Replace `torch` install with a pinned CUDA wheel matching your target GPU/CUDA version.
# For example, on CUDA 12.1 you might install a matching `torch` wheel. Modal provides
# CUDA-enabled base images and recommended wheels; adjust as needed for H100/A100.
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    torchaudio \
    librosa \
    soundfile \
    accelerate \
    huggingface-hub \
    numpy \
    scipy

WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["python3", "run_distill.py"]
