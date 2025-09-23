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
    tripo: object          # TripoSR pipeline (lazy type)

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
        cfg.rmbg_model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return model

def load_t2i(cfg: GenConfig, device: torch.device):
    # Flux.1-schnell works via diffusers AutoPipelineForText2Image
    dtype = torch.float16 if (device.type == "cuda" and cfg.t2i_dtype.lower()=="fp16") else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.t2i_model_id,
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_clip(device: torch.device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
