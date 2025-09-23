# 3DGen Service

Fast 3D Gaussian Splat generation service for mining / RunPod.

## Endpoints
- `POST /generate/` → returns `.ply` Gaussian splats (baseline miner compatible).
- `POST /preview.png` → returns PNG preview.

## Run locally
```bash
uvicorn app.main:app --reload
