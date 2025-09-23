from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv(override=True)

def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default

@dataclass
class GenConfig:
    # Text-to-image
    t2i_model_id: str = _env("T2I_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    t2i_steps: int = int(_env("T2I_STEPS", "4"))
    t2i_guidance: float = float(_env("T2I_GUIDANCE", "0.0"))
    t2i_height: int = int(_env("T2I_HEIGHT", "512"))
    t2i_width: int  = int(_env("T2I_WIDTH", "512"))
    t2i_dtype: str  = _env("T2I_DTYPE", "fp16")

    # Background removal
    rmbg_model_id: str = _env("RMBG_MODEL_ID", "briaai/RMBG-1.4")

    # 3D reconstruction
    tripo_model_id: str = _env("TRIPO_MODEL_ID", "stabilityai/TripoSR")

    # Splats
    splat_samples: int  = int(_env("SPLAT_SAMPLES", "150000"))
    splat_opacity: float = float(_env("SPLAT_OPACITY", "0.85"))

    # Validation
    max_retries: int = int(_env("MAX_RETRIES", "1"))
    clip_threshold: float = float(_env("CLIP_THRESHOLD", "0.18"))

    # General
    device_preference: str = _env("DEVICE", "auto")
    seed: int = int(_env("SEED", "0"))
    timeout_s: int = int(_env("TIMEOUT_S", "28"))

    # Auth / cache
    hf_token: str | None = _env("HF_TOKEN", None)
    hf_home: str = _env("HF_HOME", "/models/hf")
    torch_home: str = _env("TORCH_HOME", "/models/torch")

    # Debug
    fast_debug: bool = _env("FAST_DEBUG", "0") == "1"

def get_config() -> GenConfig:
    cfg = GenConfig()
    os.environ["HF_HOME"] = cfg.hf_home
    os.environ["TRANSFORMERS_CACHE"] = cfg.hf_home
    os.environ["TORCH_HOME"] = cfg.torch_home
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
    return cfg
