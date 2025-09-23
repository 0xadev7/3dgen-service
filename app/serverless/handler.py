from __future__ import annotations
import runpod
import base64
from ..config import get_config
from ..models import load_models
from ..pipeline import GaussianProcessor

# Preload
CFG = get_config()
MODELS = None if CFG.fast_debug else load_models(CFG)

# Input schema:
# {
#   "prompt": "a red toy car",
#   "return": "ply" | "png"
# }
def handler(event):
    prompt = event.get("prompt", None)
    ret = event.get("return", "ply")
    if not prompt:
        return {"error": "Missing 'prompt'."}
    gp = GaussianProcessor(CFG, prompt)
    gp.train(MODELS if MODELS is not None else {
        "pipe": None, "clip_model": None, "clip_proc": None, "rmbg": None, "tripo": None
    }, 1)
    if ret == "png":
        data = gp.get_preview_png()
        return {"content_type": "image/png", "b64": base64.b64encode(data).decode("utf-8")}
    else:
        # ply
        import io
        buffer = io.BytesIO()
        gp.get_gs_model().save_ply(buffer)
        buffer.seek(0)
        data = buffer.getvalue()
        return {"content_type": "application/octet-stream", "b64": base64.b64encode(data).decode("utf-8")}

runpod.serverless.start({"handler": handler})
