from __future__ import annotations

import base64
import io
import os
import traceback
import threading
from typing import Any, Dict, Tuple
import logging

import runpod

# Local imports (keep your package layout)
from ..config import get_config
from ..models import load_models
from ..pipeline import GaussianProcessor

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for very detailed timings
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# -----------------------
# Globals / init guards
# -----------------------
CFG = None
MODELS = None

_INIT_DONE = False
_INIT_LOCK = threading.Lock()


# -----------------------
# Helpers
# -----------------------
def _maybe_configure_hf_cache() -> None:
    """
    If running on RunPod serverless with a Network Volume,
    steer Hugging Face caches to the persistent volume.
    """
    if os.path.isdir("/runpod-volume"):
        os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
        # Optional (uncomment if you use them elsewhere):
        # os.environ.setdefault("HF_HUB_CACHE", "/runpod-volume/hf/hub")
        # os.environ.setdefault("TRANSFORMERS_CACHE", "/runpod-volume/hf/transformers")


def _extract_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod usually passes {"input": {...}}; fall back to top-level for flexibility.
    """
    if (
        isinstance(event, dict)
        and "input" in event
        and isinstance(event["input"], dict)
    ):
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


def _success_response(
    content_type: str, data: bytes, filename: str | None = None
) -> Dict[str, Any]:
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
# Robust one-time init
# -----------------------
def _init_once() -> None:
    """
    Idempotent, thread-safe init. Safe to call from both the SDK lifecycle hook and the handler.
    """
    global _INIT_DONE, CFG, MODELS
    if _INIT_DONE:
        return

    with _INIT_LOCK:
        if _INIT_DONE:  # double-checked locking
            return

        print("[init] starting...")
        try:
            _maybe_configure_hf_cache()
            CFG = get_config()

            # Respect your fast_debug behavior: skip heavy preload
            if getattr(CFG, "fast_debug", False):
                MODELS = None
                print("[init] fast_debug=True -> skipping model preload.")
            else:
                try:
                    MODELS = load_models(CFG)
                    print("[init] Models preloaded.")
                except Exception as e:
                    MODELS = None
                    print(f"[init] Model preload failed: {e}\n{traceback.format_exc()}")
                    print("[init] Proceeding with lazy/minimal models in handler.")

            _INIT_DONE = True
            print("[init] ready.")
        except Exception as e:
            # Do NOT set _INIT_DONE True on fatal error; allow handler to retry
            print(f"[init] fatal error: {e}\n{traceback.format_exc()}")
            raise


# -----------------------
# SDK lifecycle hook
# -----------------------
def init():
    """
    Called once per worker start in serverless mode.
    If the SDK ever skips this, handler() also calls _init_once().
    """
    _init_once()


# -----------------------
# Handler
# -----------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Runpod handler. Always ensures initialization before processing.
    """
    # Ensure init even if lifecycle hook wasn't called
    try:
        _init_once()
    except Exception:
        return _error_response("Initialization failed", status=500)

    try:
        data = _extract_input(event)
        prompt, ret = _validate_inputs(data)

        # Use preloaded models if available; otherwise minimal placeholders
        models = (
            MODELS
            if MODELS is not None
            else {
                "pipe": None,
                "clip_model": None,
                "clip_proc": None,
                "rmbg": None,
                "tripo": None,
            }
        )

        gp = GaussianProcessor(CFG, prompt)

        # Train for 1 step as in the original code
        gp.train(models, 1)

        if ret == "png":
            png_bytes = gp.get_preview_png()
            return _success_response("image/png", png_bytes, filename="preview.png")

        # Default: PLY
        buf = io.BytesIO()
        gp.get_gs_model().save_ply(buf)
        return _success_response(
            "application/octet-stream", buf.getvalue(), filename="model.ply"
        )

    except ValueError as ve:
        return _error_response(str(ve), status=400)
    except Exception as e:
        print(f"[handler] Unexpected error: {e}\n{traceback.format_exc()}")
        return _error_response("Internal error during generation.", status=500)


# -----------------------
# Start RunPod serverless
# -----------------------
runpod.serverless.start({"handler": handler, "init": init})
