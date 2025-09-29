import os, base64, tempfile, json, argparse
import io
import time
import runpod
from typing import Tuple, Dict, Any

import requests
from PIL import Image

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


def _persist_source_image(img, return_base64: bool) -> Dict[str, str]:
    """
    Save the source image (PNG) to disk, and either upload to S3 (if configured and return_base64 is False)
    or return as base64. Returns a small payload describing the image.
    """
    # 1) Save to a real file on disk (so you can inspect later if needed)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp_file.name
    try:
        img.save(tmp_file, format="PNG")
    finally:
        tmp_file.close()

    # 2) Also keep bytes in memory for upload / base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # 3) Prefer S3 URL if configured and caller asked for non-base64 result
    if not return_base64:
        url = _maybe_upload_bytes(png_bytes, "source.png", "image/png")
        if url:
            return {
                "filename": os.path.basename(tmp_path),
                "content_type": "image/png",
                "local_path": tmp_path,  # on-disk file path
                "url": url,
            }

    # 4) Fallback/base64 path
    return {
        "filename": os.path.basename(tmp_path),
        "content_type": "image/png",
        "local_path": tmp_path,  # on-disk file path
        "data_b64": _b64_bytes(png_bytes),
    }


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


def _load_input_image(inp) -> Image.Image:
    """
    Load a PIL image from one of: input['image_b64'] | input['image_url'].
    Returns None if not provided.
    """
    b64 = inp.get("image_b64")
    url = inp.get("image_url")
    if b64:
        try:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image_b64: {e}")
    if url:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return None


def _generate_mesh_or_from_image(
    prompt: str,
    seed: int,
    steps: int,
    image_size: int,
    use_sdxl: bool,
    lora_paths,
    lora_scales,
    return_base64: bool,
    img_in: Image.Image,
    mc_res: int,
    vertex_color: bool,
) -> Tuple[Any, Dict[str, str]]:
    """
    Use a provided source image if given; otherwise synthesize one from text.
    Then convert to mesh via TripoSR.
    """
    pipe = _ensure_pipe(use_sdxl, lora_paths, lora_scales)

    if img_in is None:
        # Text -> image
        img = pipe.text_to_image(prompt, seed=seed, steps=steps, image_size=image_size)
    else:
        img = img_in

    # Persist the source image *before* turning it into a mesh
    source_image_info = _persist_source_image(img, return_base64)

    mesh, _outs = pipe.image_to_mesh(img, mc_res=mc_res, vertex_color=vertex_color)
    return mesh, source_image_info


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
        "mode": "generate" | "generate_video" | "generate_png"
                 | "image_to_mesh" | "image_to_png" | "image_to_video",

        # Provide either a prompt (for text->image) OR an image (for image->mesh)
        "prompt": "a wooden chair",
        "image_b64": "<...>",            # optional
        "image_url": "https://...jpg",   # optional

        # Text->image knobs (ignored if image_* provided)
        "seed": 0,
        "steps": 4,
        "image_size": 512,
        "use_sdxl": false,
        "lora_paths": [],
        "lora_scales": [],

        # TripoSR knobs
        "mc_res": 384,
        "vertex_color": true,

        # Video knobs
        "seconds": 3.0,
        "fps": 16,

        # Output
        "return_base64": true            # if false and S3 is configured -> returns URL
      }
    """
    inp = job["input"]
    mode = str(inp.get("mode", "generate")).lower()

    prompt = inp.get("prompt", "")
    seed = int(inp.get("seed", 0))
    steps = int(inp.get("steps", 2))
    size = int(inp.get("image_size", 512))
    use_sdxl = bool(inp.get("use_sdxl", False))
    lora_paths = inp.get("lora_paths", [])
    lora_scales = inp.get("lora_scales", [])
    return_b64 = bool(inp.get("return_base64", True))

    # Optional direct image input
    img_in = _load_input_image(inp)

    # TripoSR controls
    mc_res = int(inp.get("mc_res", int(os.environ.get("TRIPOSR_MC_RES", "256"))))
    vertex_color = bool(
        inp.get(
            "vertex_color",
            os.environ.get("TRIPOSR_VERTEX_COLOR", "1") not in ("0", "false", "False"),
        )
    )

    t_start = time.time()

    # --- direct image->PLY ---
    if mode == "image_to_mesh":
        if img_in is None:
            return {"error": "image_to_mesh requires 'image_b64' or 'image_url'."}
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in,
            mc_res,
            vertex_color,
        )
        buf = io.BytesIO()
        mesh.export(buf, file_type="ply")
        blob = buf.getvalue()
        log.info(f"PLY size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        out = _return_payload(blob, "mesh.ply", "application/octet-stream", return_b64)
        out["source_image"] = source_image_info
        return out

    # --- direct image->PNG ---
    if mode == "image_to_png":
        if img_in is None:
            return {"error": "image_to_png requires 'image_b64' or 'image_url'."}
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in,
            mc_res,
            vertex_color,
        )
        png_bytes = render_png_bytes(mesh, size=size)
        log.info(
            f"PNG size={len(png_bytes)/1e6:.3f} MB; total={time.time()-t_start:.2f}s"
        )
        out = _return_payload(png_bytes, "preview.png", "image/png", return_b64)
        out["source_image"] = source_image_info
        return out

    # --- direct image->MP4 ---
    if mode == "image_to_video":
        if img_in is None:
            return {"error": "image_to_video requires 'image_b64' or 'image_url'."}
        seconds = float(inp.get("seconds", 3.0))
        fps = int(inp.get("fps", 16))
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in,
            mc_res,
            vertex_color,
        )
        with tempfile.TemporaryDirectory() as td:
            mp4_path = os.path.join(td, "preview.mp4")
            spin_preview(mesh, seconds=seconds, fps=fps, out_path=mp4_path, size=size)
            blob = open(mp4_path, "rb").read()
        log.info(f"MP4 size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        out = _return_payload(blob, "preview.mp4", "video/mp4", return_b64)
        out["source_image"] = source_image_info
        return out

    # --- PLY endpoint (text->image->mesh, legacy) ---
    if mode == "generate":
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in=None,  # force text->image path
            mc_res=mc_res,
            vertex_color=vertex_color,
        )
        buf = io.BytesIO()
        # If your validator requires ASCII, pass encoding="ascii"
        mesh.export(buf, file_type="ply")
        blob = buf.getvalue()
        log.info(f"PLY size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        out = _return_payload(blob, "mesh.ply", "application/octet-stream", return_b64)
        # Attach the source image info as an extra top-level field
        out["source_image"] = source_image_info
        return out

    # --- MP4 endpoint (text->image->mesh, legacy) ---
    if mode == "generate_video":
        seconds = float(inp.get("seconds", 3.0))
        fps = int(inp.get("fps", 16))
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in=None,  # force text->image path
            mc_res=mc_res,
            vertex_color=vertex_color,
        )
        with tempfile.TemporaryDirectory() as td:
            mp4_path = os.path.join(td, "preview.mp4")
            spin_preview(mesh, seconds=seconds, fps=fps, out_path=mp4_path, size=size)
            blob = open(mp4_path, "rb").read()
        log.info(f"MP4 size={len(blob)/1e6:.3f} MB; total={time.time()-t_start:.2f}s")
        out = _return_payload(blob, "preview.mp4", "video/mp4", return_b64)
        out["source_image"] = source_image_info
        return out

    # --- PNG endpoint (text->image->mesh, legacy) ---
    if mode == "generate_png":
        mesh, source_image_info = _generate_mesh_or_from_image(
            prompt,
            seed,
            steps,
            size,
            use_sdxl,
            lora_paths,
            lora_scales,
            return_b64,
            img_in=None,  # force text->image path
            mc_res=mc_res,
            vertex_color=vertex_color,
        )
        png_bytes = render_png_bytes(mesh, size=size)
        log.info(
            f"PNG size={len(png_bytes)/1e6:.3f} MB; total={time.time()-t_start:.2f}s"
        )
        out = _return_payload(png_bytes, "preview.png", "image/png", return_b64)
        out["source_image"] = source_image_info
        return out

    # Fallback / unknown mode
    return {
        "error": f"Unknown mode='{mode}'. Expected one of: "
        f"generate, generate_video, generate_png, image_to_mesh, image_to_png, image_to_video."
    }


runpod.serverless.start({"handler": run})

if __name__ == "__main__":
    # Simple local test (no Runpod broker)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=[
            "generate",
            "generate_video",
            "generate_png",
            "image_to_mesh",
            "image_to_png",
            "image_to_video",
        ],
    )
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--image_path", type=str, default="")
    ap.add_argument("--return_base64", action="store_true")
    ap.add_argument("--mc_res", type=int, default=384)
    args = ap.parse_args()

    # Optional local image loader
    image_b64 = ""
    if args.image_path and os.path.isfile(args.image_path):
        with open(args.image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

    out = run(
        {
            "input": {
                "mode": args.mode,
                "prompt": args.prompt,
                "image_b64": image_b64 if image_b64 else None,
                "return_base64": args.return_base64,
                "mc_res": args.mc_res,
            }
        }
    )
    print(json.dumps(out)[:2000] + "...")
