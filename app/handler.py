import os, base64, tempfile, json, argparse
import io
import time
import runpod

from app.pipeline import TextTo3DPipeline, render_png_from_mesh, render_png_bytes
from app.logutil import log
from app.render import spin_preview

# Optional S3 uploads for large outputs (set AWS_* envs to enable)
USE_S3 = all(
    os.getenv(k)
    for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_BUCKET",
    )
)
if USE_S3:
    import boto3

    S3 = boto3.client("s3")
    S3_BUCKET = os.environ["S3_BUCKET"]
    S3_PREFIX = os.environ.get("S3_PREFIX", "results")

# Lazily-initialized pipeline
PIPE = None


def _b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return _b64_bytes(f.read())


def _maybe_upload_bytes(blob: bytes, key_suffix: str, content_type: str):
    """
    If S3 is configured, upload and return a URL; otherwise return None.
    """
    if not USE_S3:
        return None
    # Build a unique-ish key
    ts = int(time.time() * 1000)
    key = f"{S3_PREFIX}/{ts}/{key_suffix.lstrip('/')}"
    S3.put_object(Bucket=S3_BUCKET, Key=key, Body=blob, ContentType=content_type)
    # If your bucket is public, this URL is directly downloadable:
    url = (
        f"https://{S3_BUCKET}.s3.{os.environ['AWS_DEFAULT_REGION']}.amazonaws.com/{key}"
    )
    return url


def _ensure_pipe(use_sdxl: bool, lora_paths, lora_scales):
    global PIPE
    needs_xl = bool(use_sdxl)
    has_xl = PIPE is not None and getattr(PIPE, "is_xl", False)
    if PIPE is None or (needs_xl != has_xl):
        log.info("Initializing pipeline for request...")
        PIPE = TextTo3DPipeline(use_sdxl=needs_xl)
    # Apply runtime LoRAs (optional)
    try:
        if lora_paths:
            PIPE._apply_loras(
                PIPE.sd,
                lora_paths,
                lora_scales if isinstance(lora_scales, list) else [],
            )
    except Exception as e:
        log.error(f"Runtime LoRA apply failed: {e}")
    return PIPE


def _generate_mesh(
    prompt: str,
    seed: int,
    steps: int,
    image_size: int,
    use_sdxl: bool,
    lora_paths,
    lora_scales,
):
    pipe = _ensure_pipe(use_sdxl, lora_paths, lora_scales)
    img = pipe.text_to_image(prompt, seed=seed, steps=steps, image_size=image_size)
    mesh, _outs = pipe.image_to_mesh(img)
    return mesh


def _return_payload(blob: bytes, filename: str, content_type: str, return_base64: bool):
    """
    Uniform return format for queue-mode serverless.
    If S3 is configured and return_base64 is False, upload and return URL; otherwise return base64.
    """
    if not return_base64:
        url = _maybe_upload_bytes(blob, filename, content_type)
        if url:
            return {"filename": filename, "content_type": content_type, "url": url}
        # Fallback to base64 if S3 not available
    return {
        "filename": filename,
        "content_type": content_type,
        "data_b64": _b64_bytes(blob),
    }


def run(job):
    """
    Input schema (examples):
      {
        "mode": "generate" | "generate_video" | "generate_png",
        "prompt": "a wooden chair",
        "seed": 0,
        "steps": 2,
        "image_size": 512,
        "use_sdxl": false,
        "lora_paths": [],
        "lora_scales": [],
        "seconds": 3.0,            # for video
        "fps": 16,                 # for video
        "return_base64": true      # if false and S3 is configured -> returns URL
      }
    """
    inp = job["input"]
    mode = str(inp.get("mode", "generate")).lower()

    prompt = inp["prompt"]
    seed = int(inp.get("seed", 0))
    steps = int(inp.get("steps", 2))
    size = int(inp.get("image_size", 512))
    use_sdxl = bool(inp.get("use_sdxl", False))
    lora_paths = inp.get("lora_paths", [])
    lora_scales = inp.get("lora_scales", [])
    return_b64 = bool(inp.get("return_base64", True))

    t_start = time.time()

    # --- PLY endpoint (like your FastAPI /generate) ---
    if mode == "generate":
        mesh = _generate_mesh(
            prompt, seed, steps, size, use_sdxl, lora_paths, lora_scales
        )
        buf = io.BytesIO()
        # If your validator requires ASCII, pass encoding="ascii"
        mesh.export(buf, file_type="ply")
        blob = buf.getvalue()
        log.info(f"PLY size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        return _return_payload(blob, "mesh.ply", "application/octet-stream", return_b64)

    # --- MP4 endpoint (like your /generate_video) ---
    if mode == "generate_video":
        seconds = float(inp.get("seconds", 3.0))
        fps = int(inp.get("fps", 16))
        mesh = _generate_mesh(
            prompt, seed, steps, size, use_sdxl, lora_paths, lora_scales
        )
        with tempfile.TemporaryDirectory() as td:
            mp4_path = os.path.join(td, "preview.mp4")
            spin_preview(mesh, seconds=seconds, fps=fps, out_path=mp4_path, size=size)
            blob = open(mp4_path, "rb").read()
        log.info(f"MP4 size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        return _return_payload(blob, "preview.mp4", "video/mp4", return_b64)

    # --- PNG endpoint (easy to paste elsewhere) ---
    if mode == "generate_png":
        mesh = _generate_mesh(
            prompt, seed, steps, size, use_sdxl, lora_paths, lora_scales
        )
        png_bytes = render_png_bytes(mesh, size=size)
        log.info(
            f"PNG size={len(png_bytes)/1e6:.3f} MB; total={time.time()-t_start:.2f}s"
        )
        return _return_payload(png_bytes, "preview.png", "image/png", return_b64)

    # Fallback / unknown mode
    return {
        "error": f"Unknown mode='{mode}'. Expected one of: generate, generate_video, generate_png."
    }


runpod.serverless.start({"handler": run})

if __name__ == "__main__":
    # Simple local test (no Runpod broker)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "generate_video", "generate_png"],
    )
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--return_base64", action="store_true")
    args = ap.parse_args()
    out = run(
        {
            "input": {
                "mode": args.mode,
                "prompt": args.prompt,
                "return_base64": args.return_base64,
            }
        }
    )
    print(json.dumps(out)[:500] + "...")
