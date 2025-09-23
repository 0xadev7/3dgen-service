import io, time
from fastapi import FastAPI, Form, Depends, Response, HTTPException
from omegaconf import OmegaConf
from .deps import get_config, get_models
from .gaussian_processor import GaussianProcessor
from .logging_conf import setup_logging

app = FastAPI(title="3DGen Service", version="1.0.0")
setup_logging(app)

@app.post("/generate/")
async def generate(
    prompt: str = Form(...),
    opt: OmegaConf = Depends(get_config),
    models = Depends(get_models),
) -> Response:
    t0 = time.time()
    gp = GaussianProcessor(opt, prompt)
    gp.train(models, getattr(opt, "iters", None))
    t1 = time.time()
    app.logger.info(f"Generation took: {(t1 - t0) / 60.0:.2f} min")

    buf = io.BytesIO()
    gp.get_gs_model().save_ply(buf)
    buf.seek(0)
    data = buf.getbuffer()
    if len(data) == 0:
        # validation failed across attempts → empty result (ignored by miner)
        return Response(content=b"", media_type="application/octet-stream")

    t2 = time.time()
    app.logger.info(f"Saving/encoding took: {(t2 - t1) / 60.0:.2f} min")
    return Response(data, media_type="application/octet-stream")

@app.post("/preview.png")
async def preview(
    prompt: str = Form(...),
    opt: OmegaConf = Depends(get_config),
    models = Depends(get_models),
):
    gp = GaussianProcessor(opt, prompt)
    gp.train(models, getattr(opt, "iters", None))
    png_path = gp.preview_png()
    try:
        with open(png_path, "rb") as f:
            png_bytes = f.read()
        return Response(png_bytes, media_type="image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Preview rendering failed.")
