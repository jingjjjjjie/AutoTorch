FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Make python point to python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1 build)
RUN pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

# Install OverLoCK dependencies
RUN pip install \
    natten==0.17.1+torch230cu121 \
    -f https://shi-labs.com/natten/wheels/ \
    --trusted-host shi-labs.com

RUN pip install "setuptools==65.5.0" && \
    pip install \
    timm==0.6.12 \
    mmengine==0.2.0 \
    einops

# Copy repo and test script
COPY OverLoCK/ /workspace/OverLoCK/
COPY test_forward.py /workspace/test_forward.py

# Verify imports (cuda will be False at build time, True at runtime with --gpus)
RUN python -c "import torch; print('torch:', torch.__version__)" && \
    python -c "import natten; print('natten OK')" && \
    python -c "import timm; print('timm:', timm.__version__)" && \
    python -c "import mmengine; print('mmengine:', mmengine.__version__)"