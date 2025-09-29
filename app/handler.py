import os, base64, tempfile, json, argparse
import time
import runpod

from app.pipeline import TextTo3DPipeline, render_png_from_mesh
from app.logutil import log
from app.render import spin_preview

# Lazily-initialized pipeline
PIPE = None


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run(job):
    inp = job["input"]
    prompt = inp["prompt"]
    seed = inp.get("seed", 0)
    steps = int(inp.get("steps", 2))
    size = int(inp.get("image_size", 512))
    multiview = bool(inp.get("multiview", False))
    use_sdxl = bool(inp.get("use_sdxl", False))
    lora_paths = inp.get("lora_paths", [])
    lora_scales = inp.get("lora_scales", [])
    mv_steps = int(inp.get("multiview_steps", 1))
    mv_size = int(inp.get("multiview_image_size", 384))
    clip_topk = int(inp.get("clip_topk", 2))
    rot_sec = float(inp.get("rotation_seconds", 3.0))
    rot_fps = int(inp.get("rotation_fps", 16))
    return_b64 = bool(inp.get("return_base64", False))

    global PIPE
    # Init or swap pipeline based on SDXL flag
    needs_xl = use_sdxl
    has_xl = PIPE is not None and getattr(PIPE, "is_xl", False)
    if PIPE is None or (needs_xl != has_xl):
        log.info("Initializing pipeline for request...")
        PIPE = TextTo3DPipeline(use_sdxl=use_sdxl)
        # Apply runtime LoRAs if provided
        try:
            if lora_paths:
                PIPE._apply_loras(
                    PIPE.sd,
                    lora_paths,
                    lora_scales if isinstance(lora_scales, list) else [],
                )
        except Exception as e:
            log.error(f"Runtime LoRA apply failed: {e}")

    t_start = time.time()
    with tempfile.TemporaryDirectory() as td:
        # Stage 1: text→image / multiview
        if multiview:
            raw = PIPE.multiview_images(
                prompt, seed=seed, steps=mv_steps, image_size=mv_size
            )
            ranked = PIPE.rank_views_with_clip(prompt, raw, topk=clip_topk)
            candidates = []
            for tag, im, score in ranked:
                m, _o = PIPE.image_to_mesh(im)
                candidates.append((f"{tag} (clip={score:.3f})", m))
            from app.render import choose_best_mesh

            tag, mesh = choose_best_mesh(candidates)
        else:
            img = PIPE.text_to_image(prompt, seed=seed, steps=steps, image_size=size)
            mesh, _outs = PIPE.image_to_mesh(img)

        # Save mesh
        glb_path = os.path.join(td, "mesh.glb")
        ply_path = os.path.join(td, "mesh.ply")
        mesh.export(glb_path)
        mesh.export(
            ply_path
        )  # change to encoding="ascii" if your validator requires ASCII PLY

        # Single-frame PNG preview + optional spin MP4
        png_path = os.path.join(td, "preview.png")
        render_png_from_mesh(mesh, png_path, size=size)

        mp4_path = os.path.join(td, "preview.mp4")
        spin_preview(mesh, seconds=rot_sec, fps=rot_fps, out_path=mp4_path, size=size)

        if return_b64:
            out = {
                "preview_png_b64": _b64(png_path),
                "preview_mp4_b64": _b64(mp4_path),
                "mesh_glb_b64": _b64(glb_path),
                "mesh_ply_b64": _b64(ply_path),
            }
            log.info(f"job done in {time.time()-t_start:.2f}s")
            return out
        else:
            out = {
                "preview_png": png_path,
                "preview_mp4": mp4_path,
                "mesh_glb": glb_path,
                "mesh_ply": ply_path,
            }
            log.info(f"job done in {time.time()-t_start:.2f}s")
            return out


runpod.serverless.start({"handler": run})

if __name__ == "__main__":
    # Simple local test
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, help="prompt")
    args = ap.parse_args()
    if args.test:
        out = run({"input": {"prompt": args.test, "return_base64": False}})
        print(json.dumps(out, indent=2))
