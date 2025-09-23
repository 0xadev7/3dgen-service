# app/pipeline/bg_remove.py
import numpy as np
from PIL import Image

def cut_foreground(rmbg_model, img_pil, opt):
    if img_pil.mode != "RGBA":
        img_pil = img_pil.convert("RGBA")
    mask = rmbg_model.predict_mask(img_pil)  # float [0..1], shape (h,w)
    im = np.array(img_pil).astype(np.uint8)
    if im.shape[2] == 3:  # if somehow RGB, add alpha
        a = (mask * 255).astype(np.uint8)
        rgba = np.dstack([im, a])
    else:
        a = (mask * 255).astype(np.uint8)
        rgba = im.copy()
        rgba[:, :, 3] = a
    return Image.fromarray(rgba, mode="RGBA")
