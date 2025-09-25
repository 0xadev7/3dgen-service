FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
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

# Pre-cache models to avoid HF downloads at runtime

# 1) FLUX (private-ready)
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN python3 - <<'PY'
import os
from huggingface_hub import login
from diffusers import FluxPipeline

tok = os.getenv("HF_TOKEN")
if tok:
    try:
        login(tok)
    except Exception as e:
        print("HF login warning:", e)

FluxPipeline.from_pretrained(
    os.getenv("FLUX_MODEL_ID","black-forest-labs/FLUX.1-schnell"),
    token=tok
)
print("cached FLUX")
PY

# 2) RMBG (private-ready)
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN python3 - <<'PY'
import os
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModelForImageSegmentation

tok = os.getenv("HF_TOKEN")
if tok:
    try:
        login(tok)
    except Exception as e:
        print("HF login warning:", e)

AutoImageProcessor.from_pretrained("briaai/RMBG-1.4", token=tok)
AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", token=tok)
print("cached RMBG")
PY

# 3) TripoSR (private-ready)
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN python3 - <<'PY'
import os
from huggingface_hub import login
from tsr.system import TSR

tok = os.getenv("HF_TOKEN")
if tok:
    try:
        login(tok)
    except Exception as e:
        print("HF login warning:", e)

TSR.from_pretrained(os.getenv("TRIPOSR_MODEL_ID","stabilityai/TripoSR"), token=tok)
print("cached TripoSR")
PY

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

CMD ["python3", "-m", "app.serverless"]
