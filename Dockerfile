FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY runpod.toml ./runpod.toml
COPY README.md ./README.md
COPY scripts ./scripts

# Pre-cache HF models to avoid cold downloads at runtime.
RUN python3 - <<'PY'
from diffusers import FluxPipeline
FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
print("cached FLUX")
PY

RUN python3 - <<'PY'
from transformers import AutoImageProcessor, AutoModelForImageSegmentation
AutoImageProcessor.from_pretrained("briaai/RMBG-1.4")
AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4")
print("cached RMBG")
PY

RUN python3 - <<'PY'
from tsr.system import TSR
TSR.from_pretrained("stabilityai/TripoSR")
print("cached TripoSR")
PY

RUN python3 -m app.sanity_check

ENV LOG_LEVEL=INFO
ENV RMBG_MODE=torch
ENV FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
ENV TRIPOSR_MODEL_ID=stabilityai/TripoSR

CMD ["python3", "-m", "app.serverless"]
