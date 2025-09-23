# app/models/bg/briai_rmbg.py
import io
import os
from typing import Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort


_MODEL_REPO = "briaai/RMBG-1.4"
_MODEL_FILE = "model.onnx"
_INPUT_SIZE = (1024, 1024)  # BRIA RMBG default resolution (HxW)


def _letterbox_rgba(img: Image.Image, size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Pads the image to 'size' with transparent pixels, keeping aspect ratio.
    Returns: padded_rgb (H,W,3 float32 0..1), original (w,h), (offset_x, offset_y)
    """
    w0, h0 = img.size
    H, W = size
    scale = min(W / w0, H / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    img_resized = img.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ox = (W - nw) // 2
    oy = (H - nh) // 2
    canvas.paste(img_resized, (ox, oy))
    # To RGB for the model (alpha used only as hint; RMBG model is trained on RGB)
    rgb = np.array(canvas.convert("RGB")).astype(np.float32) / 255.0
    return rgb, (w0, h0), (ox, oy)


class BriaRMBGOnnx:
    def __init__(self, onnx_path: str, use_cuda: bool = True):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict_mask(self, img_pil: Image.Image) -> np.ndarray:
        """
        Args:
            img_pil: RGB or RGBA PIL image
        Returns:
            mask float32 in [0,1] with original image resolution (h, w)
        """
        if img_pil.mode not in ("RGB", "RGBA"):
            img_pil = img_pil.convert("RGBA")
        # Keep alpha if present; RMBG is robust without it too.
        rgb_in, (w0, h0), (ox, oy) = _letterbox_rgba(img_pil.convert("RGBA"), _INPUT_SIZE)
        # NCHW
        x = np.transpose(rgb_in, (2, 0, 1))[None, ...].astype(np.float32)

        # RMBG expects [0,1] normalized RGB; no mean/std needed
        out = self.sess.run([self.output_name], {self.input_name: x})[0]  # (1,1,H,W)
        pred = out[0, 0]  # (H, W), 0..1

        # Remove letterbox, resize back to original size
        H, W = _INPUT_SIZE
        crop = pred  # (H, W)
        # Un-pad
        left, top = ox, oy
        right, bottom = left + int(round(w0 * min(W / w0, H / h0))), top + int(round(h0 * min(W / w0, H / h0)))
        crop = crop[top:bottom, left:right]
        # Back to original resolution
        crop_pil = Image.fromarray((crop * 255).astype(np.uint8), mode="L").resize((w0, h0), Image.LANCZOS)
        mask = np.asarray(crop_pil).astype(np.float32) / 255.0
        return np.clip(mask, 0.0, 1.0)


def load_rmbg(cfg):
    """
    Downloads BRIA RMBG v1.4 ONNX from Hugging Face and loads ONNXRuntime (GPU if available).
    """
    onnx_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
    use_cuda = (getattr(cfg, "device", "cpu") == "cuda")
    return BriaRMBGOnnx(onnx_path, use_cuda=use_cuda)
