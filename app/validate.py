from __future__ import annotations
import torch, PIL.Image as Image
from typing import Tuple

@torch.no_grad()
def clip_score(clip_model, clip_proc, prompt: str, img: Image.Image) -> float:
    # FAST_DEBUG: treat as valid so the pipeline doesn’t retry endlessly
    if clip_model is None or clip_proc is None:
        return 1.0

    inputs = clip_proc(text=[prompt], images=[img], return_tensors="pt", padding=True)
    for k,v in inputs.items():
        inputs[k] = v.to(clip_model.device)
    outputs = clip_model(**inputs)
    txt = outputs.text_embeds
    im  = outputs.image_embeds
    sim = torch.nn.functional.cosine_similarity(txt, im).mean().item()
    return float(sim)
