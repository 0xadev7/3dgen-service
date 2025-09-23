# Minimal, rasterizes points with covariance as screen-space splats (approx).
# For higher fidelity, plug-in a proper gsplat renderer.

from PIL import Image
import numpy as np

def render_views(ply_path, views=4):
    # Load points/colors; set up simple orbit cams; call render_gs_preview per view
    # Return list of PIL.Image
    ims = []
    for i in range(views):
        out_png = ply_path.replace("model.ply", f"view_{i}.png")
        render_gs_preview(ply_path, out_png)
        ims.append(Image.open(out_png).convert("RGB"))
    return ims

def render_gs_preview(ply_path, out_png, opt=None):
    # Placeholder: project points as disks; for speed and simplicity.
    # You can later swap in a CUDA splat renderer (gsplat) for fidelity.
    from .utils import load_ply_points
    pts, cols, cov = load_ply_points(ply_path)
    if len(pts) == 0:
        Image.new("RGB", (640, 640), (30,30,30)).save(out_png); return
    W = H = 768
    img = np.zeros((H,W,3), dtype=np.float32)
    # naive orthographic projection
    p = pts.copy()
    p -= p.mean(0, keepdims=True)
    scale = 0.45 / (np.max(np.abs(p)) + 1e-6)
    p = p * scale + 0.5
    xs = np.clip((p[:,0]*W).astype(int), 0, W-1)
    ys = np.clip((p[:,1]*H).astype(int), 0, H-1)
    for (x,y), c in zip(zip(xs,ys), cols):
        img[y, x] = np.maximum(img[y,x], c/255.0)
    Image.fromarray((img*255).astype(np.uint8)).save(out_png)
