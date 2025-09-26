# CUDA 12.1.1 runtime with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid Python building from source; install minimal OS deps
ENV DEBIAN_FRONTEND=noninteractive     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     HF_HOME=/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git build-essential python3-dev ninja-build \
    libgl1 libegl1 libglib2.0-0 libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90"
ENV MAX_JOBS=4

# Create a virtualenv to keep image tidy
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy files
WORKDIR /workspace
COPY requirements.txt ./

# Install Core pytorch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2

# Vendor TripoSR (no setup.py/pyproject)
RUN git clone --depth=1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR
RUN pip install --no-cache-dir -r /opt/TripoSR/requirements.txt
ENV PYTHONPATH=/opt/TripoSR:$PYTHONPATH

# torchmcubes & app dependencies
RUN pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Using vendored TripoSR"

# Cache directories for models
RUN mkdir -p /weights && chmod -R 777 /weights

# Copy app
COPY app ./app
COPY README.md ./README.md

# Sanity Check
RUN python app/sanity_check.py || true

# (Optional) Warm lightweight tokenizer/configs at build without pulling huge checkpoints
# This keeps image size smaller; full weights are cached on first request into /weights
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYOPENGL_PLATFORM=egl
ENV HF_HOME=/runpod-volume/weights

RUN mkdir -p /runpod-volume/weights && chmod -R 777 /runpod-volume || true && chmod -R 777 /runpod-volume/weights || true

# Expose none; Runpod Serverless uses handler entrypoint
CMD [ "python", "-m", "app.handler" ]
