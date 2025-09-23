# Python 3.10 + Torch 2.3.1 + CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps
RUN apt-get update && apt-get install -y \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install Python deps
COPY requirements.txt .
# torch/torchvision/torchaudio are ALREADY in the base image; don't reinstall
RUN sed -i '/^torch\(\|\[.*\]\)\|^torchvision\|^torchaudio/d' requirements.txt && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# App code
COPY . .
RUN chmod +x start.sh

EXPOSE 8000
ENV GEN_CONFIG=configs/default.yaml
CMD ["./start.sh"]
