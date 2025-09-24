from __future__ import annotations
import torch, os
from typing import TypedDict
from .config import GenConfig
from diffusers import AutoPipelineForText2Image
from transformers import CLIPModel, CLIPProcessor


class Models(TypedDict):
    pipe: AutoPipelineForText2Image
    clip_model: CLIPModel
    clip_proc: CLIPProcessor
    rmbg: torch.nn.Module  # BRIA RMBG
    tripo: object  # TripoSR pipeline (lazy type)


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    # auto:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rmbg(cfg: GenConfig, device: torch.device):
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        cfg.rmbg_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model


def load_t2i(cfg: "GenConfig", device: torch.device):
    """
    Fast-loading Flux.1-schnell (or similar T2I) pipeline with:
      - sentencepiece available (Dockerfile)
      - xFormers attention if present
      - TF32 on Ampere+
      - QKV fusion if supported
      - channels_last layouts
      - disabled progress bar
    """
    use_fp16 = device.type == "cuda" and cfg.t2i_dtype.lower() == "fp16"
    dtype = torch.float16 if use_fp16 else torch.float32

    # Build the pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.t2i_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME"),
        trust_remote_code=True,  # harmless if not required; fixes some custom repos
    )

    # ---- Speed knobs ----
    if device.type == "cuda":
        # Allow TF32 on Ampere+ for extra throughput with negligible quality impact
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Enable memory-efficient attention (biggest speed win if xformers is present)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers not installed or unsupported; ignore
            pass

        # Fuse QKV projections when available (diffusers >= 0.30)
        if hasattr(pipe, "fuse_qkv_projections"):
            try:
                pipe.fuse_qkv_projections()
            except Exception:
                pass

        # channels_last can help UNet/Transformer throughput on CUDA
        try:
            pipe.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass
        # Some pipelines expose a "transformer" module as well
        try:
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                pipe.transformer.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # (Optional) Disable safety checker if present for a tiny speed bump
    try:
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = lambda images, **kwargs: (images, False)
    except Exception:
        pass

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def load_clip(device: torch.device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_proc


def load_models(cfg: GenConfig) -> Models:
    device = pick_device(cfg.device_preference)
    pipe = load_t2i(cfg, device)
    rmbg = load_rmbg(cfg, device)
    clip_model, clip_proc = load_clip(device)
    return {
        "pipe": pipe,
        "clip_model": clip_model,
        "clip_proc": clip_proc,
        "rmbg": rmbg,
    }
