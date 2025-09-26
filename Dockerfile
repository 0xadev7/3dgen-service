# ─────────────────────────────────────────────────────────────────────────────
# Base image: CUDA 12.8.x is required for Blackwell (sm_120) GPUs like RTX 5090
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    # Silence only transformers' FutureWarnings in logs (optional)
    PYTHONWARNINGS="ignore:.*TRANSFORMERS_CACHE.*:FutureWarning"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3.10-dev python3.10-venv \
    python3-pip git curl ca-certificates build-essential ninja-build git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Latest pip toolchain
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Speed up native builds (if any)
RUN pip install --no-cache-dir --upgrade cmake ninja scikit-build-core pybind11

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch NIGHTLY for CUDA 12.8 with sm_120 support
# (stable cu124 wheels DO NOT support sm_120)
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --pre --upgrade \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision

WORKDIR /workspace
COPY requirements.txt ./requirements.txt

# Clone TripoSR and its deps; add to PYTHONPATH
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR \
 && pip install --no-cache-dir -r /opt/TripoSR/requirements.txt
ENV PYTHONPATH=/opt/TripoSR:$PYTHONPATH

# Install project deps
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY app ./app
COPY runpod.toml ./runpod.toml
COPY README.md ./README.md
COPY scripts ./scripts

# Build-time sanity check
RUN python3 -m app.sanity_check

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration
# ─────────────────────────────────────────────────────────────────────────────
ENV LOG_LEVEL=INFO
ENV RMBG_MODE=torch
ENV FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
ENV TRIPOSR_MODEL_ID=stabilityai/TripoSR

# Use the RunPod Network Volume as HF cache
ENV HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf \
    DIFFUSERS_CACHE=/runpod-volume/hf/diffusers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
# NOTE: Make sure your RunPod template or job does NOT set TRANSFORMERS_CACHE.

# Entry script
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Default command
CMD ["/usr/local/bin/entrypoint.sh"]
