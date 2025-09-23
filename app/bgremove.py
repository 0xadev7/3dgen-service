from __future__ import annotations
import torch
from .config import GenConfig
import PIL.Image as Image
import numpy as np

@torch.no_grad()
def remove_bg(model, img: Image.Image, cfg: GenConfig) -> Image.Image:
    # BRIA RMBG expects normalized tensor
    device = next(model.parameters()).device
    img_rgb = img.convert("RGB")
    im = np.array(img_rgb).astype(np.float32) / 255.0
    x = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).to(device)
    # forward
    pred = model(x)[0]  # (B,1,H,W) with trust_remote_code
    pred = torch.sigmoid(pred)
    m = pred[0,0].detach().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    m3 = np.stack([m,m,m], axis=-1)
    cut = (np.array(img_rgb).astype(np.float32) * m3).astype(np.uint8)
    out = Image.fromarray(cut, "RGB")
    return out
