from __future__ import annotations
import torch
from typing import Tuple
from .config import GenConfig
import PIL.Image as Image
import numpy as np

@torch.no_grad()
def text2image(pipe, prompt: str, cfg: GenConfig, seed: int) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    img = pipe(
        prompt=prompt,
        num_inference_steps=cfg.t2i_steps,
        guidance_scale=cfg.t2i_guidance,
        height=cfg.t2i_height,
        width=cfg.t2i_width,
        generator=generator
    ).images[0]
    return img

def fake_image_for_debug(prompt: str, cfg: GenConfig) -> Image.Image:
    # CPU-friendly synthetic image
    w, h = cfg.t2i_width, cfg.t2i_height
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Write a prompt hash stripe so it’s deterministic
    stripe = abs(hash(prompt)) % 255
    arr[:, : w//2, 0] = stripe
    arr[:, w//2 :, 1] = (stripe * 2) % 255
    return Image.fromarray(arr, "RGB")
