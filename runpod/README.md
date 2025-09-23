# RunPod Integration

- Use `runpod/handler.py` for Serverless entrypoint.
- Or expose `app/main.py` via uvicorn in your pod.
- Set env var `GEN_CONFIG=configs/fast.yaml` for big GPUs.
