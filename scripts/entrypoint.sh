#!/usr/bin/env bash
set -euo pipefail

# Allow override (fallbacks to /runpod-volume)
ROOT="${VOLUME_ROOT:-/runpod-volume}"

export HF_HOME="${HF_HOME:-$ROOT/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-$HF_HOME/diffusers}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

echo "[entrypoint] ROOT=$ROOT"
echo "[entrypoint] HF_HOME=$HF_HOME"
mount | grep -E " $ROOT " || echo "[entrypoint] WARNING: volume not mounted at $ROOT"

# Optional: one-time warmup (safe to remove if you prefer lazy)
LOCK="$HF_HOME/.warm.lock"
if [ ! -f "$LOCK" ]; then
  ( umask 022; : > "$LOCK" ) || true
  python3 - <<'PY' || true
import os
from huggingface_hub import login
from diffusers import FluxPipeline
from transformers import AutoImageProcessor, AutoModelForImageSegmentation
from tsr.system import TSR

tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if tok:
    try: login(tok); print("[warm] HF login OK")
    except Exception as e: print("[warm] HF login warn:", e)

flux_id = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
rmbg_id = os.getenv("RMBG_MODEL_ID", "briaai/RMBG-1.4")
trip_id = os.getenv("TRIPOSR_MODEL_ID", "stabilityai/TripoSR")

for fn, name in [
    (lambda t: FluxPipeline.from_pretrained(flux_id, token=t), "FLUX"),
    (lambda t: (AutoImageProcessor.from_pretrained(rmbg_id, token=t),
                AutoModelForImageSegmentation.from_pretrained(rmbg_id, token=t)), "RMBG"),
    (lambda t: TSR.from_pretrained(trip_id, token=t), "TripoSR"),
]:
    try:
        fn(tok); print(f"[warm] Cached {name}")
    except Exception as e:
        print(f"[warm] {name} prefetch failed: {e}")
PY
fi

exec python3 -m app.serverless
