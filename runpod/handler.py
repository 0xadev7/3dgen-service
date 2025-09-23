import runpod
from app.deps import get_config, get_models
from app.gaussian_processor import GaussianProcessor

cfg = get_config()
models = get_models()

def handler(event):
    prompt = event.get("input", {}).get("prompt", "")
    if not prompt:
        return {"error": "Missing prompt"}
    gp = GaussianProcessor(cfg, prompt)
    gp.train(models, getattr(cfg, "iters", None))
    ply_path = gp.get_gs_model().ply_path
    # return as base64 or S3 URL in your infra; here we just return path
    return {"ply_path": ply_path}

runpod.serverless.start({"handler": handler})
