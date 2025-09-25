FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3.10-dev python3.10-venv \
    python3-pip git curl ca-certificates build-essential ninja-build git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade cmake ninja scikit-build-core pybind11
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0

# Clone TripoSR and its deps; add to PYTHONPATH
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR \
    && pip install --no-cache-dir -r /opt/TripoSR/requirements.txt
ENV PYTHONPATH=/opt/TripoSR:$PYTHONPATH

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY runpod.toml ./runpod.toml
COPY README.md ./README.md
COPY scripts ./scripts

# Build-time sanity check
RUN python3 -m app.sanity_check

ENV LOG_LEVEL=INFO
ENV RMBG_MODE=torch
ENV FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
ENV TRIPOSR_MODEL_ID=stabilityai/TripoSR

# Default to /runpod-volume; endpoint env will override if needed
ENV HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf/transformers \
    DIFFUSERS_CACHE=/runpod-volume/hf/diffusers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Entry script to warm cache (optional) and start worker
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
CMD ["/usr/local/bin/entrypoint.sh"]
