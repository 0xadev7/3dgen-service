# recon.py
from __future__ import annotations
import os, tempfile, subprocess, sys, glob, shutil
from typing import Optional, Sequence
import numpy as np
import torch
import PIL.Image as Image
import trimesh

TRIPOSR_DIR = os.environ.get("TRIPOSR_DIR", "/opt/TripoSR")  # set in Dockerfile

def _find_mesh_file(out_dir: str) -> str:
    # Prefer common mesh formats; adjust if you expect others.
    patterns: Sequence[str] = ("*.obj", "*.ply", "*.glb", "*.gltf")
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(out_dir, pat)))
        if files:
            return files[0]
    raise FileNotFoundError(f"No mesh file found in {out_dir!r}")

def _run_triposr_cli(img_path: str, out_dir: str, model_id: Optional[str] = None,
                     extra_args: Optional[Sequence[str]] = None,
                     timeout_sec: Optional[int] = 600) -> None:
    """
    Calls TripoSR's CLI: python /opt/TripoSR/run.py <image> --output-dir <out_dir> [--model <hf_repo>]
    Add other flags here if you customize your fork.
    """
    run_py = os.path.join(TRIPOSR_DIR, "run.py")
    if not os.path.isfile(run_py):
        raise RuntimeError(f"TripoSR run.py not found at {run_py}. Is TRIPOSR_DIR set correctly?")
    cmd = [sys.executable, run_py, img_path, "--output-dir", out_dir]
    if model_id:
        cmd.extend(["--model", model_id])
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True, timeout=timeout_sec)

@torch.no_grad()
def image_to_mesh(
    tripo: Optional[object],              # kept for API compatibility; ignored (pass anything or None)
    img: Image.Image,
    *,
    model_id: Optional[str] = None,       # e.g. "stabilityai/TripoSR"
    extra_args: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None # kept for API compatibility; TripoSR CLI picks GPU automatically
) -> trimesh.Trimesh:
    """
    Converts a PIL image to a normalized trimesh via TripoSR's CLI.
    - No Python import of tripoSR required.
    - Writes a temp PNG, runs /opt/TripoSR/run.py, loads the first mesh in the output dir.
    """

    # FAST_DEBUG path preserved: return a simple sphere when tripo is falsy and you want a stub
    if tripo is None and os.environ.get("TRIPO_FAST_DEBUG", "0") == "1":
        return trimesh.creation.icosphere(subdivisions=2, radius=0.5)

    tmp_dir = tempfile.mkdtemp(prefix="triposr_")
    try:
        inp = os.path.join(tmp_dir, "input.png")
        img.convert("RGBA").save(inp)

        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        _run_triposr_cli(inp, out_dir, model_id=model_id, extra_args=extra_args)

        mesh_path = _find_mesh_file(out_dir)
        mesh = trimesh.load(mesh_path, force="mesh")

        # Normalize to unit cube (same as your original)
        mesh = mesh.copy()
        if not mesh.is_watertight:
            mesh.remove_unreferenced_vertices()
        bbox = mesh.bounds
        center = (bbox[0] + bbox[1]) / 2.0
        scale = float(np.max(bbox[1] - bbox[0]))
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / max(scale, 1e-6))
        return mesh
    finally:
        # keep artifacts by setting TRIPO_KEEP_TMP=1
        if os.environ.get("TRIPO_KEEP_TMP", "0") != "1":
            shutil.rmtree(tmp_dir, ignore_errors=True)
