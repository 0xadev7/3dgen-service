import os, torch
from .utils import get_logger, timed

logger = get_logger()

class Text2Img:
    def __init__(self, device="cuda", dtype=torch.float16):
        with timed(logger, "load_flux"):
            from diffusers import FluxPipeline
            model_id = os.getenv("FLUX_MODEL_ID","black-forest-labs/FLUX.1-schnell")
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=os.getenv("HF_HOME", None),
                variant=None
            )
            self.pipe.to(device)
            self.device = device
            self.dtype = dtype
            logger.info("FLUX loaded", extra={"extra": {"model_id": model_id}})

    @torch.inference_mode()
    def __call__(self, prompt: str, seed: int, steps: int = 4, width=640, height=640, guidance=1.0):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with timed(logger, "flux_generate"):
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                width=width, height=height,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        return image
