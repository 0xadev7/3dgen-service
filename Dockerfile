FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .
RUN chmod +x start.sh

# warm weights during build (optional): keep off to speed image build
# RUN python3 scripts/warmup.py

EXPOSE 8000
ENV GEN_CONFIG=configs/default.yaml
CMD ["./start.sh"]
