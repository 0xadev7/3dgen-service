# scripts/warmup.py
"""
Warm up all models by doing a tiny end-to-end pass:
1) Text2Image (Flux.1-schnell or SDXL-Turbo) with tiny steps/res
2) Background removal (BRIA RMBG ONNX)
3) Single-view 3D (TripoSR) to mesh
4) Mesh -> Gaussian Splats (.ply)
5) Render preview + CLIP-based validation pass

This downloads weights on first run and seeds all kernels so the first request to the API is fast.
"""

import os, sys, time, argparse, random
from contextlib import contextmanager

# Make repo root importable no matter how this file is run
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# App imports
from app.deps import get_config, get_models
from app.gaussian_processor import GaussianProcessor
from app.pipeline.text2img import generate_image
from app.pipeline.bg_remove import cut_foreground
from app.pipeline.to_3d import single_view_mesh
from app.pipeline.mesh_to_gs import mesh_to_gaussians
from app.pipeline.render import render_gs_preview
from app.pipeline.validate import validate_gs


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[warmup] {name} took {dt:.2f}s")

def _override_for_warmup(cfg, width: int, height: int, steps: int):
    # make a non-destructive shallow copy with a few overrides
    cfg = OmegaConf.merge(cfg, {
        "text2img": {
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": 1.0,
            "seed": 12345678,
        },
        "to3d": {
            "simplify_to": min(50_000, getattr(cfg.to3d, "simplify_to", 120_000)),
            "max_faces": min(80_000, getattr(cfg.to3d, "max_faces", 200_000)),
        },
        "gs": {
            "points_per_m2": min(1000, getattr(cfg.gs, "points_per_m2", 2500)),
            "max_points": min(150_000, getattr(cfg.gs, "max_points", 1_200_000)),
            "cov_scale": getattr(cfg.gs, "cov_scale", 0.35),
        },
        "validate": {
            "views": 2,
            "max_attempts": 1,
            "min_clip": getattr(cfg.validate, "min_clip", 0.28),
            "min_aesthetic": getattr(cfg.validate, "min_aesthetic", 5.4),
            "nsfw_block": getattr(cfg.validate, "nsfw_block", True),
        },
        "io": {
            "out_dir": "/tmp/3dgen_warmup"
        }
    })
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=os.environ.get("WARMUP_PROMPT", "a tiny clay teapot"))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--full", action="store_true",
                        help="Run the entire GaussianProcessor.train() once (uses validation & potential retry).")
    args = parser.parse_args()

    os.makedirs("/tmp/3dgen_warmup", exist_ok=True)

    # Load config & models
    with timer("load config"):
        base_cfg = get_config()
        cfg = _override_for_warmup(base_cfg, args.width, args.height, args.steps)

    with timer("load models"):
        models = get_models()   # loads t2i, rmbg, triposr, clip

    # Tiny random seed for determinism
    import torch
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    if args.full:
        # Run the *same* pipeline the API uses, end-to-end
        with timer("full pipeline (GaussianProcessor.train)"):
            gp = GaussianProcessor(cfg, args.prompt)
            gp.train(models, getattr(cfg, "iters", None))
            # Save artifacts
            ply_path = gp.get_gs_model().ply_path
            png_path = gp.preview_png()
            print(f"[warmup] wrote {ply_path}")
            print(f"[warmup] wrote {png_path}")
        print("[warmup] ✅ FULL warmup finished.")
        return

    # Manual staged warmup (faster than full)
    run_id = "warmup"
    out_dir = cfg.io.out_dir
    mesh_path = os.path.join(out_dir, f"{run_id}_mesh.obj")
    ply_path = os.path.join(out_dir, f"{run_id}.ply")
    png_path = os.path.join(out_dir, f"{run_id}.png")

    # 1) Text → Image
    with timer("text2img forward"):
        img = generate_image(models.t2i, args.prompt, cfg, new_seed=False)
        # ensure size (some backends may deviate slightly)
        img = img.resize((args.width, args.height), Image.LANCZOS)
        img.save(os.path.join(out_dir, f"{run_id}_rgb.jpg"))

    # 2) Background removal
    with timer("background removal"):
        fg = cut_foreground(models.rmbg, img, cfg)
        fg.save(os.path.join(out_dir, f"{run_id}_rgba.png"))

    # 3) TripoSR single-view to mesh
    with timer("TripoSR mesh recon"):
        single_view_mesh(models.triposr, fg, mesh_path, cfg)

    # 4) Mesh -> Gaussian Splats
    with timer("mesh->gaussian splats"):
        mesh_to_gaussians(mesh_path, ply_path, cfg)

    # 5) Preview render + CLIP quick validation
    with timer("preview render"):
        render_gs_preview(ply_path, png_path, cfg)
    with timer("CLIP validate"):
        ok = validate_gs(models.clip, ply_path, args.prompt, cfg)
        print(f"[warmup] validation ok? {ok}")

    print(f"[warmup] ✅ staged warmup finished. Artifacts:\n  {ply_path}\n  {png_path}")

if __name__ == "__main__":
    main()
