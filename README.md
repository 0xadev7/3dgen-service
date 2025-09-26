# threegen-runpod (Text→3D, <=30s hot path)

A minimal, **fast** text→3D serverless worker for **Runpod** that keeps the Docker image lean and uses **HF cache** to meet a ~30s *hot* runtime target on 32–48GB GPUs.

## Pipeline
1) **SD-Turbo** (or SDXL-Turbo) → a single 512×512 view from text in ~1–2 steps.
2) **TripoSR** → single-image → coarse **mesh** in seconds.
3) **Quick 360° preview** → offscreen render → MP4 + GLB/PLY export.

### Why this is fast
- Uses distilled turbo diffusion (few steps).
- TripoSR is feed-forward and GPU-accelerated.
- Lower preview FPS & resolution by default; configurable.
- Hugging Face models are cached via `HF_HOME=/weights`.

> **Note**: First cold boot will pull models into cache which can exceed 30s. Subsequent inferences re-use cache and meet the 30s target.

## Run locally (GPU)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=$PWD/weights
python -m app.handler --test "a shiny pink bicycle"
```

## Deploy on Runpod Serverless
1. Build & push image:
```bash
docker build -t <repo>/threegen-runpod:latest .
docker push <repo>/threegen-runpod:latest
```
2. Create a **Serverless** endpoint with that image, enable GPU (32–48GB), set ENV:
```
HF_HOME=/weights
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
3. Test synchronous job:
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync -H "Content-Type: application/json" -d '{
  "input": {
    "prompt": "a shiny pink bicycle",
    "seed": 42
  }
}'
```
Response returns signed URLs (or base64) for `preview_mp4` and `mesh_glb`.

## Environment
- **CUDA 12.1.1 runtime** (matches Three-Gen note)
- **PyTorch 2.1** wheels
- No source builds (keeps image small and avoids slow compiles).

## Endpoints / Schema (Runpod Worker)
- `handler.run(job)` input:
  - `prompt` *(str, required)*
  - `seed` *(int, optional)*
  - `steps` *(int, optional, default=2)*  — diffusion steps
  - `image_size` *(int, optional, default=512)*
  - `rotation_seconds` *(float, default=3.0)*
  - `rotation_fps` *(int, default=16)*
  - `return_base64` *(bool, default=false)*
- Returns: paths or base64 for `preview_mp4`, `mesh_glb`, `mesh_ply`

## Credits
- [Stability AI **SD‑Turbo**](https://huggingface.co/stabilityai/sd-turbo)
- [**TripoSR**](https://github.com/VAST-AI-Research/TripoSR)



## Plus Mode (multi-view, still ~30s hot)
Set `"multiview": true` to render 4–5 lightweight views with SD‑Turbo and pick the best TripoSR mesh via a quick heuristic. Tunables:
- `multiview_steps` (default 1)
- `multiview_image_size` (default 384)

Example:
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT>/runsync -H "Content-Type: application/json" -d '{
  "input": {"prompt":"a wooden chair", "multiview": true}
}'
```

## TripoSR install (no failing pip wheel)
We install **TripoSR from GitHub** and **torchmcubes from GitHub** inside the Dockerfile to avoid broken wheels:
```Dockerfile
pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git
pip install --no-cache-dir git+https://github.com/VAST-AI-Research/TripoSR.git
```
This follows the official repo troubleshooting guidance. If CUDA major versions mismatch, rebuild `torchmcubes` for the target container.


### CLIP re-ranker
Set `clip_topk` (default 2) with `"multiview": true` to select the most on-prompt views **before** reconstruction using `openai/clip-vit-base-patch32`:
```json
{"input":{"prompt":"a stone statue of a lion","multiview":true,"clip_topk":2}}
```
You can override the CLIP repo via `CLIP_REPO` env if you want a smaller/larger variant.


## GPU sizing
- **Recommended (fastest, roomy)**: L40S 48GB or A40 48GB.
- **Works fine**: A100 40GB / V100 32GB (ensure CUDA 12-compatible driver for this image).
- **Minimum**: 16–24GB will run single-view; multiview+CLIP is happier at ≥24GB.

## Network volume cache (HF models)
Use a Runpod **Network Volume** mounted at `/runpod-volume`. Set `HF_HOME=/runpod-volume/weights` so all Hugging Face models persist across workers and cold starts.

**Size guidance**
- SD‑Turbo + TripoSR + CLIP: ~3–5 GB after first warm.
- With SDXL‑Turbo and/or multiple LoRAs: plan **20–30 GB**.
- Recommendation: allocate **20 GB** if you plan to experiment; **10 GB** minimum for SD‑Turbo only.

**Endpoint environment**
```
HF_HOME=/runpod-volume/weights
PYOPENGL_PLATFORM=egl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## SDXL‑Turbo & LoRA
- Switch to SDXL‑Turbo per request:
```json
{"input":{"prompt":"ceramic teapot product shot","use_sdxl":true}}
```
- Load LoRAs at runtime:
```json
{"input":{
  "prompt":"a fantasy castle at sunset",
  "lora_paths":["/runpod-volume/weights/lora/castle_lora.safetensors"],
  "lora_scales":[0.8]
}}
```
Or set env:
```
LORA_PATHS=/runpod-volume/weights/lora/castle_lora.safetensors
LORA_SCALES=0.8
```

## Logging
- Structured JSON logs by default. Override with `LOG_FORMAT=plain` and `LOG_LEVEL=DEBUG`.
- Build-time sanity script prints CUDA/GL/ffmpeg diagnostics during `docker build`.


## Runpod Serverless (with Network Volume)
1) Create a **Network Volume** and attach it to the endpoint mounted at `/runpod-volume`.
2) Set environment variables:
   ```
   HF_HOME=/runpod-volume/weights
   PYOPENGL_PLATFORM=egl
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   LOG_LEVEL=INFO
   ```
3) Set **Active (min) workers = 1** to keep one hot worker and avoid cold starts.
4) (Optional) Warm the cache once by running a test request so SD/SDXL, CLIP, and TripoSR populate `/runpod-volume/weights`.
