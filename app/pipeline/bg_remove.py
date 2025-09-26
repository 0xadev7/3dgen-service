import os, torch
from PIL import Image
import numpy as np
from .utils import get_logger, timed

logger = get_logger()

class BgRemover:
    def __init__(self, device="cuda"):
        self.device = device
        self.mode = os.getenv("RMBG_MODE","torch")  # "torch" or "onnx"
        with timed(logger, "load_rmbg"):
            if self.mode == "onnx":
                import onnxruntime as ort
                repo = "briaai/RMBG-2.0"
                base = os.getenv("HF_HOME","/root/.cache/huggingface")
                self.sess = ort.InferenceSession(
                    os.path.join(base, "hub"), providers=["CUDAExecutionProvider","CPUExecutionProvider"]
                )
                self.kind = "onnx"
            else:
                from transformers import AutoModelForImageSegmentation, AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained("briaai/RMBG-2.0")
                self.model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True).to(self.device)
                self.kind = "torch"
            logger.info("RMBG loaded", extra={"extra":{"kind": self.kind}})

    @torch.inference_mode()
    def __call__(self, image: Image.Image):
        w, h = image.size
        with timed(logger, "rmbg_run"):
            if self.kind == "torch":
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                mask = out.logits.sigmoid().float().cpu().detach()[0,0].numpy()
            else:
                arr = np.array(image.convert("RGB"))
                scale = 1024 / max(h, w)
                import cv2
                resized = cv2.resize(arr, (int(w*scale), int(h*scale)))
                inp = resized.transpose(2,0,1)[None].astype(np.float32)/255.0
                mask = self.sess.run(None, {"input": inp})[0][0,0]
                mask = cv2.resize(mask, (w,h))
        mask = np.clip(mask, 0, 1)
        fg = np.dstack([np.array(image.convert("RGB")), (mask*255).astype(np.uint8)])
        return Image.fromarray(fg, mode="RGBA"), mask
