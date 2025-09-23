# 3D Image Generation (Turbo → RMBG → TripoSR → Gaussians)

Fast single-image 3D pipeline optimized for serverless on Runpod.

## Pipeline
1. **Text-to-Image**: `stabilityai/sd-turbo` (commercially safe, 1–4 steps).
2. **Background Removal**: `briaai/RMBG-1.4` (robust).
3. **3D Reconstruction**: `stabilityai/TripoSR` → normalized mesh.
4. **Gaussian Splat Export**: fast surface sampling to Gaussian PLY.
5. **Self-Validation**: CLIP similarity; optional 1 re-try (configurable).
6. **Preview**: mesh-rendered PNG for quick visual inspection.

## Endpoints
- `POST /generate/` → **PLY** (application/octet-stream)
  - Body (form): `prompt=<string>`
- `POST /preview_png/` → **PNG** preview
  - Body (form): `prompt=<string>`
- `GET /health` → `{"ok": true}`

These match the baseline miner’s expectations (`GaussianProcessor(...).train(...); get_gs_model().save_ply(...)`).

## Local (CPU) Test
CPU-friendly **debug mode** (skips heavy models, returns deterministic synthetic output but still runs the entire pipeline code-path):

```bash
export FAST_DEBUG=1        # CPU stub for quick smoke test
docker compose up --build
python -m tests.test_local
