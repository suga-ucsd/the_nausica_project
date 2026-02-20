FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Basic dev tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    tmux \
    htop \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Python goodies
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    tqdm \
    transformers \
    safetensors \
    accelerate


# RUN pip install --no-deps vllm \
#   && pip install transformers sentencepiece einops


# Set workspace
WORKDIR /workspace

COPY . .

CMD ["/bin/bash"]

