# 3D PLY Service (Runpod Serverless)

Text prompt → PLY point cloud in ≤30s using:
- FLUX.1 [schnell] (Diffusers) for fast, high-quality images
- BRIA RMBG-1.4 for robust background removal
- TripoSR for feed-forward single-view 3D reconstruction
- Trimesh → PLY

## Quickstart

```bash
# Build
docker build -t yourrepo/plysvc:latest .

# Push (Docker Hub example)
docker push yourrepo/plysvc:latest
```

Create a Runpod **Serverless Endpoint** with this image and the default command.

### Invoke (example payload)
```json
{
  "input": {
    "prompt": "a photorealistic plush dragon, studio lighting",
    "steps": 3,
    "n_points": 200000,
    "seed": 12345,
    "return_ply_b64": true
  }
}
```

### Outputs
```json
{
  "ok": true,
  "seed": 12345,
  "points": 200000,
  "ply_path": "/tmp/tmpabc123.ply",
  "ply_b64": "<base64>",
  "latency_ms": 12250
}
```

### Notes
- Uses CUDA 12.1 runtime; if your GPU image differs, pin compatible `torch/torchvision/xformers` versions.
- HF assets cached at build to avoid downloads on first request.
