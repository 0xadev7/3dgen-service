from __future__ import annotations
import os, io, logging
from time import time
from fastapi import FastAPI, Form, Depends, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import get_config, GenConfig
from .models import load_models
from .pipeline import GaussianProcessor

logger = logging.getLogger("uvicorn")
app = FastAPI(title="3D Gen (Turbo->TripoSR->Gaussians)")

# CORS for validator hits / local tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    cfg = get_config()
    app.state.cfg = cfg
    try:
        if cfg.fast_debug:
            # no model load needed
            app.state.models = {
                "pipe": None, "clip_model": None, "clip_proc": None, "rmbg": None, "tripo": None
            }
        else:
            app.state.models = load_models(cfg)
        logger.info("Models loaded.")
    except Exception as e:
        logger.exception("Failed to load models: %s", e)
        raise

def get_cfg() -> GenConfig:
    return app.state.cfg

@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    opt: GenConfig = Depends(get_cfg),
) -> Response:
    t0 = time()
    gp = GaussianProcessor(opt, prompt)
    # keep API compatible: train(models, iters)
    if opt.fast_debug:
        gp.train(app.state.models, 1)
    else:
        gp.train(app.state.models, 1)
    t1 = time()
    logger.info(f"Generation took {(t1 - t0):.2f} s")
    buffer = io.BytesIO()
    gp.get_gs_model().save_ply(buffer)
    buffer.seek(0)
    buf = buffer.getbuffer()
    t2 = time()
    logger.info(f"Saving took {(t2 - t1):.2f} s")
    return Response(buf, media_type="application/octet-stream")

@app.post("/preview_png/")
async def preview_png(
    prompt: str = Form(),
    opt: GenConfig = Depends(get_cfg),
) -> Response:
    gp = GaussianProcessor(opt, prompt)
    if opt.fast_debug:
        gp.train(app.state.models, 1)
    else:
        gp.train(app.state.models, 1)
    png = gp.get_preview_png()
    return Response(png, media_type="image/png")

@app.get("/health")
async def health():
    return {"ok": True}
