import os, time, tempfile
from PIL import Image
import numpy as np
import runpod
from app.pipeline.utils import get_logger, timed, fix_seed, to_b64_bytes
from app.pipeline.text2img import Text2Img
from app.pipeline.bg_remove import BgRemover
from app.pipeline.singleview_3d import SingleView3D
from app.pipeline.validate import quick_validate

logger = get_logger()

TEXT2IMG = None
RMBG     = None
TRIPO    = None

def init_models():
    global TEXT2IMG, RMBG, TRIPO
    if TEXT2IMG is None:
        TEXT2IMG = Text2Img()
    if RMBG is None:
        RMBG = BgRemover()
    if TRIPO is None:
        TRIPO = SingleView3D()

def handler(event):
    init_models()
    t0 = time.time()
    inp = event.get("input") or {}

    prompt   = inp.get("prompt", "a red glossy toy robot, centered, studio lighting")
    steps    = int(inp.get("steps", 3))
    width    = int(inp.get("width", 640))
    height   = int(inp.get("height", 640))
    n_points = int(inp.get("n_points", 200_000))
    seed     = fix_seed(inp.get("seed", -1))
    guidance = float(inp.get("guidance", 1.0))
    return_ply_b64 = bool(inp.get("return_ply_b64", True))

    logger.info("request_start", extra={"extra": {"prompt": prompt, "steps": steps, "w": width, "h": height, "n_points": n_points, "seed": seed}})

    with timed(logger, "pipeline_total"):
        img = TEXT2IMG(prompt, seed, steps=steps, width=width, height=height, guidance=guidance)
        rgba, mask = RMBG(img)
        mesh = TRIPO.to_mesh(rgba)
        pts, colors = TRIPO.mesh_to_pointcloud(mesh, n_points=n_points)

        ok, reasons = quick_validate(mask, pts)
        if not ok:
            logger.warning("validation_failed", extra={"extra": {"reasons": reasons}})
            return {
                "ok": False,
                "reasons": reasons,
                "latency_ms": int((time.time()-t0)*1000)
            }

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_path = tmp.name
        TRIPO.write_ply(pts, colors, ply_path)

    out = {
        "ok": True,
        "seed": seed,
        "points": len(pts),
        "ply_path": ply_path,
        "latency_ms": int((time.time()-t0)*1000)
    }
    if return_ply_b64:
        out["ply_b64"] = to_b64_bytes(ply_path)
    logger.info("request_end", extra={"extra": {"ok": True, "latency_ms": out["latency_ms"]}})
    return out

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
