from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import PIL.Image as Image
import torch

from .config import GenConfig

logger = logging.getLogger(__name__)

# Many diffusion backends want H/W to be multiples of 8 (or 64).
MULTIPLE = 8
MIN_SIZE = 64
MAX_SIZE = 4096  # generous ceiling; adjust for your models


def _round_to_multiple(x: int, k: int = MULTIPLE) -> int:
    return max(k, (x + k - 1) // k * k)


def _sanitize_size(h: int, w: int) -> Tuple[int, int]:
    """Clamp to sane bounds and round to model-friendly multiple."""
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size HxW={h}x{w}")
    h = min(max(h, MIN_SIZE), MAX_SIZE)
    w = min(max(w, MIN_SIZE), MAX_SIZE)
    return _round_to_multiple(h), _round_to_multiple(w)


@torch.no_grad()
def text2image(pipe, prompt: str, cfg: GenConfig, seed: int) -> Image.Image:
    """
    Generate one image from text using a Diffusers-like pipeline.

    - Robust seed handling (uses pipeline device if available).
    - Validates and auto-adjusts H/W to multiples of 8.
    - Uses half precision autocast on CUDA for speed/memory (fallbacks to fp32).
    - Passes through optional negative prompt from cfg if present.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("`prompt` must be a non-empty string.")

    # Resolve size (sanitized & rounded)
    h_cfg = getattr(cfg, "t2i_height", 512)
    w_cfg = getattr(cfg, "t2i_width", 512)
    height, width = _sanitize_size(h_cfg, w_cfg)
    if (height, width) != (h_cfg, w_cfg):
        logger.debug(
            "text2image: adjusted size from %dx%d to %dx%d to satisfy model constraints",
            h_cfg,
            w_cfg,
            height,
            width,
        )

    # Resolve steps / guidance with safe defaults
    steps = int(getattr(cfg, "t2i_steps", 30))
    guidance = float(getattr(cfg, "t2i_guidance", 5.0))
    negative_prompt = getattr(cfg, "t2i_negative_prompt", None)

    # Device & dtype
    device = getattr(
        pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch_dtype = (
        torch.float16
        if (
            hasattr(torch, "cuda")
            and device
            and getattr(device, "type", None) == "cuda"
        )
        else torch.float32
    )

    # Seed generator tied to pipeline device
    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        # Some generators don’t allow device placement; fall back to CPU
        generator = torch.Generator().manual_seed(int(seed))

    # Optional kwargs for different pipelines (don’t break if unsupported)
    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "generator": generator,
    }
    if negative_prompt is not None:
        call_kwargs["negative_prompt"] = negative_prompt

    # Autocast only on CUDA float16-capable devices
    use_autocast = torch_dtype == torch.float16
    try:
        if use_autocast:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = pipe(**call_kwargs)
        else:
            out = pipe(**call_kwargs)
    except TypeError as e:
        # Some pipelines don’t accept height/width (e.g., fixed-size or latent inputs)
        logger.debug("text2image: retrying without explicit height/width due to: %s", e)
        call_kwargs.pop("height", None)
        call_kwargs.pop("width", None)
        if use_autocast:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = pipe(**call_kwargs)
        else:
            out = pipe(**call_kwargs)

    # Extract image
    try:
        img = out.images[0]  # Diffusers standard
    except Exception as e:
        raise RuntimeError(
            f"Pipeline output does not contain images: {type(out)!r}"
        ) from e

    if not isinstance(img, Image.Image):
        # Some pipelines return numpy arrays; convert to PIL
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8), mode="RGB")
        else:
            raise TypeError(f"First image is not a PIL.Image (got {type(img)!r})")

    return img


def fake_image_for_debug(prompt: str, cfg: GenConfig) -> Image.Image:
    """
    CPU-only synthetic image for fast debugging; deterministic w.r.t. prompt.
    """
    w = int(getattr(cfg, "t2i_width", 512))
    h = int(getattr(cfg, "t2i_height", 512))
    h, w = _sanitize_size(h, w)

    arr = np.zeros((h, w, 3), dtype=np.uint8)

    # Deterministic stripes from prompt hash
    stripe = abs(hash(prompt)) % 255
    arr[:, : w // 2, 0] = stripe
    arr[:, w // 2 :, 1] = (stripe * 2) % 255

    return Image.fromarray(arr, "RGB")
