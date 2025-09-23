from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Tuple

import runpod

# Local imports (keep your package layout)
from ..config import get_config
from ..models import load_models
from ..pipeline import GaussianProcessor


# -----------------------
# Globals (initialized in init())
# -----------------------
CFG = None
MODELS = None


# -----------------------
# Helpers
# -----------------------
def _maybe_configure_hf_cache() -> None:
    """
    If running on RunPod serverless with a Network Volume,
    steer Hugging Face caches to the persistent volume.
    """
    if os.path.isdir("/runpod-volume"):
        # Keep it simple; you can also set HF_HUB_CACHE/TRANSFORMERS_CACHE if you want
        os.environ.setdefault("HF_HOME", "/runpod-volume/hf")


def _extract_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod usually passes {"input": {...}}; fall back to top-level for flexibility.
    """
    if isinstance(event, dict) and "input" in event and isinstance(event["input"], dict):
        return event["input"]
    return event


def _validate_inputs(data: Dict[str, Any]) -> Tuple[str, str]:
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Missing required field 'prompt' (string).")

    ret = data.get("return", "ply")
    if ret not in ("ply", "png"):
        raise ValueError("Invalid 'return'. Must be 'ply' or 'png'.")

    return prompt, ret


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _success_response(content_type: str, data: bytes, filename: str | None = None) -> Dict[str, Any]:
    resp = {
        "ok": True,
        "content_type": content_type,
        "b64": _b64(data),
    }
    if filename:
        resp["filename"] = filename
    return resp


def _error_response(message: str, status: int = 400) -> Dict[str, Any]:
    return {"ok": False, "error": message, "status": status}


# -----------------------
# Lifecycle
# -----------------------
def init():
    """
    Called once per worker start. Preload config and (optionally) models.
    """
    global CFG, MODELS

    _maybe_configure_hf_cache()

    CFG = get_config()
    try:
        # Respect your original fast_debug behavior
        if getattr(CFG, "fast_debug", False):
            MODELS = None
            print("[init] fast_debug=True -> skipping model preload.")
        else:
            MODELS = load_models(CFG)
            print("[init] Models preloaded.")
    except Exception as e:
        # If preload fails, keep MODELS=None; handler() will create the minimal stubs
        MODELS = None
        print(f"[init] Model preload failed: {e}. Will proceed with lazy/minimal models.")


# -----------------------
# Handler
# -----------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = _extract_input(event)
        prompt, ret = _validate_inputs(data)

        # Prepare models: use preloaded if present, otherwise minimal stubs
        models = MODELS if MODELS is not None else {
            "pipe": None, "clip_model": None, "clip_proc": None, "rmbg": None, "tripo": None
        }

        gp = GaussianProcessor(CFG, prompt)
        # 1 training step as in the original code
        gp.train(models, 1)

        if ret == "png":
            png_bytes = gp.get_preview_png()
            # Optional name; UI may use it for saving
            return _success_response("image/png", png_bytes, filename="preview.png")

        # Default: PLY
        buf = io.BytesIO()
        gp.get_gs_model().save_ply(buf)
        ply_bytes = buf.getvalue()
        return _success_response("application/octet-stream", ply_bytes, filename="model.ply")

    except ValueError as ve:
        return _error_response(str(ve), status=400)
    except Exception as e:
        # Avoid leaking internals; print full details to logs, return compact error to caller
        print(f"[handler] Unexpected error: {e}")
        return _error_response("Internal error during generation.", status=500)


# Start RunPod serverless
runpod.serverless.start({"handler": handler, "init": init})
