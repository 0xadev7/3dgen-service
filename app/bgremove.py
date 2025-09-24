from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import PIL.Image as Image

from .config import GenConfig

logger = logging.getLogger(__name__)

EPS = 1e-6


def _extract_pred_mask(obj: Any) -> Tensor:
    """
    Try to extract a mask tensor from various model output formats.
    Expected final shape: B×C×H×W (C>=1). We'll later select channel 0.
    """
    # Direct tensor
    if isinstance(obj, torch.Tensor):
        return obj

    # List / Tuple: take the first tensor-ish thing
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        # Some models return [tensor] or (tensor, ...)
        return _extract_pred_mask(obj[0])

    # Dict: try common keys
    if isinstance(obj, dict):
        for key in ("logits", "masks", "mask", "out", "saliency"):
            if key in obj:
                return _extract_pred_mask(obj[key])

    raise TypeError(f"Unsupported model output type for mask extraction: {type(obj)!r}")


def _ensure_bchw(x: Tensor) -> Tensor:
    """
    Normalize tensor to B×C×H×W float32.
    Accepts H×W, H×W×C, B×H×W, B×C×H×W, etc.
    """
    if x.dim() == 2:
        # H×W -> 1×1×H×W
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        # Could be H×W×C or C×H×W
        if x.shape[0] in (1, 3):  # C×H×W
            x = x.unsqueeze(0)  # B×C×H×W
        else:
            # H×W×C -> B×C×H×W
            x = x.permute(2, 0, 1).unsqueeze(0)
    elif x.dim() == 4:
        # Already B×C×H×W
        pass
    else:
        raise ValueError(
            f"Unsupported mask tensor dim={x.dim()}, shape={tuple(x.shape)}"
        )

    # Float for downstream ops
    if not torch.is_floating_point(x):
        x = x.float()
    return x


def _maybe_sigmoid(x: Tensor) -> Tensor:
    """
    Apply sigmoid if values look like logits (range not in [0,1]).
    Heuristic: if any value < 0 - 1e-3 or > 1 + 1e-3, we treat as logits.
    """
    if x.min() < -1e-3 or x.max() > 1 + 1e-3:
        return torch.sigmoid(x)
    return x.clamp(0.0, 1.0)


def _resize_bilinear(x: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    """
    Resize B×C×H×W to target (H, W) via bilinear.
    """
    return torch.nn.functional.interpolate(
        x, size=size_hw, mode="bilinear", align_corners=False
    )


@torch.no_grad()
def remove_bg(model: Any, img: Image.Image, cfg: GenConfig) -> Image.Image:
    """
    Background removal that tolerates various model output formats.
    Returns an RGB image with background attenuated by the predicted mask.
    """
    # FAST_DEBUG or intentionally skipped model load: pass-through
    if model is None:
        logger.debug("remove_bg: model is None; pass-through")
        return img.convert("RGB")

    # Device & dtype
    try:
        device = next(model.parameters()).device  # works for nn.Module
    except Exception:
        # Fallback: keep CPU if model doesn't expose parameters (rare)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("remove_bg: fallback device=%s", device)

    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    im_np = np.asarray(img_rgb, dtype=np.float32) / 255.0  # H×W×3

    # To tensor (B×C×H×W)
    x: Tensor = torch.from_numpy(im_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Forward
    pred_raw = model(x)
    try:
        pred = _extract_pred_mask(pred_raw)
    except Exception as e:
        logger.exception("remove_bg: failed to extract mask from model output")
        raise

    pred = pred.to(device)
    pred = _ensure_bchw(pred)

    # Use first channel as foreground prob/logit
    if pred.shape[1] > 1:
        logger.debug("remove_bg: multi-channel mask detected, using channel 0")
    m = pred[:, 0:1, :, :]  # B×1×H×W

    # Resize to image size if needed
    if (m.shape[-1] != w) or (m.shape[-2] != h):
        logger.debug(
            "remove_bg: resizing mask from %sx%s to %sx%s",
            m.shape[-1],
            m.shape[-2],
            w,
            h,
        )
        m = _resize_bilinear(m, (h, w))

    # Map to [0,1]
    m = _maybe_sigmoid(m)
    # Normalize robustly in case it's near-constant
    m_min, m_max = m.amin(dim=(-2, -1), keepdim=True), m.amax(
        dim=(-2, -1), keepdim=True
    )
    m = (m - m_min) / (m_max - m_min + EPS)
    m = m.clamp(0.0, 1.0)

    # Broadcast to 3 channels and to CPU numpy
    m3 = m.repeat(1, 3, 1, 1)[0].permute(1, 2, 0).contiguous().cpu().numpy()  # H×W×3

    cut = im_np * m3
    cut = (cut * 255.0).round().astype(np.uint8)
    out = Image.fromarray(cut, mode="RGB")

    logger.debug(
        "remove_bg: done (mask min=%.4f max=%.4f mean=%.4f)",
        float(m.min()),
        float(m.max()),
        float(m.mean()),
    )
    return out
