from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import PIL.Image as Image

logger = logging.getLogger(__name__)


@torch.no_grad()
def clip_score(clip_model, clip_proc, prompt: str, img: Image.Image) -> float:
    """
    Return a text–image similarity score using a CLIP-like model/processor.

    - If either model or processor is None, returns 1.0 (fast-debug path).
    - Sends inputs to the model's device and uses autocast on CUDA if applicable.
    - Works with newer Transformers (outputs.text_embeds/image_embeds)
      and falls back to .get_text_features() / .get_image_features() otherwise.
    - L2-normalizes embeddings before cosine similarity.
    """
    # FAST_DEBUG: treat as valid so the pipeline doesn’t retry endlessly
    if clip_model is None or clip_proc is None:
        return 1.0

    device = getattr(
        clip_model,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    use_fp16 = device.type == "cuda"

    # Processor handles tokenization & image prep
    inputs: Dict[str, Any] = clip_proc(
        text=[prompt],
        images=[img],
        return_tensors="pt",
        padding=True,
    )

    # Move all tensors to model device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)

    # Try the standard forward first
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    try:
        if use_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = clip_model(**inputs)
        else:
            outputs = clip_model(**inputs)

        txt = getattr(outputs, "text_embeds", None)
        im = getattr(outputs, "image_embeds", None)

        if txt is None or im is None:
            # Fallback to explicit feature extraction (API compatible across many versions)
            text_kwargs = {}
            image_kwargs = {}

            # Common processor output keys
            if "input_ids" in inputs:
                text_kwargs["input_ids"] = inputs["input_ids"]
            if "attention_mask" in inputs:
                text_kwargs["attention_mask"] = inputs["attention_mask"]
            if "pixel_values" in inputs:
                image_kwargs["pixel_values"] = inputs["pixel_values"]

            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    txt = clip_model.get_text_features(**text_kwargs)
                    im = clip_model.get_image_features(**image_kwargs)
            else:
                txt = clip_model.get_text_features(**text_kwargs)
                im = clip_model.get_image_features(**image_kwargs)

    except Exception as e:
        logger.exception("clip_score: model forward failed")
        # Conservative fallback: consider it a low but non-zero match
        return 0.0

    # Ensure tensors and shapes
    if not (isinstance(txt, torch.Tensor) and isinstance(im, torch.Tensor)):
        raise TypeError("CLIP model did not return tensor embeddings.")
    if txt.ndim == 1:
        txt = txt.unsqueeze(0)
    if im.ndim == 1:
        im = im.unsqueeze(0)

    # L2-normalize then cosine similarity == dot product
    txt = _normalize(txt.float())
    im = _normalize(im.float())

    # If batching >1, average similarity
    sim = (txt * im).sum(dim=-1).mean().item()

    # Map from [-1,1] → [0,1] if you prefer a [0,1] score. Keeping raw cosine is fine too.
    # sim = 0.5 * (sim + 1.0)

    return float(sim)
