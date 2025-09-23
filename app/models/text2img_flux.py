from diffusers import FluxPipeline, StableDiffusionXLPipeline
import torch

def load_text2img(cfg):
    if cfg.text2img.backend == "flux":
        # flux.1-schnell
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if cfg.precision == "fp16" else torch.float32
        )
    elif cfg.text2img.backend == "sdxl_turbo":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if cfg.precision == "fp16" else torch.float32
        )
    else:
        raise ValueError(f"Unknown backend {cfg.text2img.backend}")
    pipe.to(cfg.device)
    pipe.enable_attention_slicing()
    return pipe
