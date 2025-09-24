from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
import trimesh

logger = logging.getLogger(__name__)

TRIPOSR_DIR = os.environ.get("TRIPOSR_DIR", "/opt/TripoSR")  # set in Dockerfile
DEFAULT_TIMEOUT_S = int(os.environ.get("TRIPO_TIMEOUT_S", "600"))  # override if needed


# -----------------------------
# Helpers
# -----------------------------
def _find_mesh_file(out_dir: str) -> str:
    """
    Pick the first mesh file among common formats, preferring OBJ/PLY.
    """
    patterns: Sequence[str] = ("*.obj", "*.ply", "*.glb", "*.gltf")
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(out_dir, pat)))
        if files:
            logger.debug("mesh candidate selected: %s", files[0])
            return files[0]
    raise FileNotFoundError(f"No mesh file found in {out_dir!r} (patterns={patterns})")


def _run_triposr_cli(
    img_path: str,
    out_dir: str,
    model_id: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    timeout_sec: Optional[int] = None,
) -> None:
    """
    Call TripoSR's CLI:
      python /opt/TripoSR/run.py <image> --output-dir <out_dir> [--model <hf_repo>] [...]
    Captures stdout/stderr for easier debugging.
    """
    run_py = os.path.join(TRIPOSR_DIR, "run.py")
    if not os.path.isfile(run_py):
        raise RuntimeError(f"TripoSR run.py not found at {run_py}. Set TRIPOSR_DIR correctly?")

    cmd = [sys.executable, run_py, img_path, "--output-dir", out_dir]
    if model_id:
        cmd.extend(["--model", model_id])
    if extra_args:
        cmd.extend(extra_args)

    timeout = timeout_sec or DEFAULT_TIMEOUT_S
    logger.info("TripoSR CLI start", extra={"cmd": " ".join(cmd), "timeout_s": timeout})

    # Inherit minimal env, allow CUDA selection, etc.
    env = os.environ.copy()

    try:
        res = subprocess.run(
            cmd,
            check=False,
            timeout=timeout,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        logger.error("TripoSR CLI timed out after %ss", timeout)
        raise TimeoutError(f"TripoSR CLI timed out after {timeout}s") from e

    if res.returncode != 0:
        logger.error(
            "TripoSR CLI failed (code=%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
            res.returncode,
            res.stdout[-4000:],  # tail for brevity
            res.stderr[-4000:],
        )
        raise RuntimeError(
            f"TripoSR CLI failed (code={res.returncode}). "
            f"See logs for details. Stderr tail:\n{res.stderr[-800:]}"
        )
    else:
        logger.debug("TripoSR CLI stdout tail:\n%s", res.stdout[-2000:])
        logger.info("TripoSR CLI completed successfully.")


def _as_trimesh(obj: Union[trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    """
    Convert trimesh.Scene to a single Trimesh by concatenation if needed.
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj

    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("Loaded scene contains no geometries.")
        if len(geoms) == 1:
            return geoms[0]
        # Concatenate multiple geometries into one
        try:
            merged = trimesh.util.concatenate(geoms)
        except Exception:
            # Fallback: merge via summation
            merged = geoms[0].copy()
            for g in geoms[1:]:
                merged = trimesh.util.concatenate([merged, g])
        return merged

    raise TypeError(f"Unsupported trimesh object type: {type(obj)!r}")


def _normalize_to_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center mesh and scale to fit inside a unit cube (~[-0.5, 0.5]^3).
    """
    mesh = mesh.copy()
    try:
        if not mesh.is_watertight:
            mesh.remove_unreferenced_vertices()
    except Exception:
        # Some meshes might not support these ops; continue anyway.
        pass

    bbox = mesh.bounds  # (2, 3)
    center = (bbox[0] + bbox[1]) / 2.0
    extent = bbox[1] - bbox[0]
    scale = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    scale = max(scale, 1e-6)

    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh


# -----------------------------
# Public API
# -----------------------------
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
    Convert a PIL image to a normalized trimesh via TripoSR's CLI.

    Flow:
      1) Save input PNG to a temp dir.
      2) Run /opt/TripoSR/run.py to produce a mesh.
      3) Load the first mesh found, normalize to unit cube, return.

    Env toggles:
      - TRIPO_FAST_DEBUG=1 : returns an icosphere (skips CLI).
      - TRIPO_KEEP_TMP=1   : keeps the temporary directory for debugging.
      - TRIPO_TIMEOUT_S    : override CLI timeout seconds (default 600).
    """
    # FAST_DEBUG path preserved
    if tripo is None and os.environ.get("TRIPO_FAST_DEBUG", "0") == "1":
        logger.info("image_to_mesh: TRIPO_FAST_DEBUG=1, returning icosphere stub.")
        return trimesh.creation.icosphere(subdivisions=2, radius=0.5)

    tmp_dir = tempfile.mkdtemp(prefix="triposr_")
    keep_tmp = os.environ.get("TRIPO_KEEP_TMP", "0") == "1"
    logger.debug("image_to_mesh: tmp_dir=%s keep_tmp=%s", tmp_dir, keep_tmp)

    try:
        inp = os.path.join(tmp_dir, "input.png")
        # Ensure RGBA to preserve any alpha (TripoSR can accept RGB as well)
        img.convert("RGBA").save(inp)
        logger.debug("Saved input image -> %s", inp)

        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # Run CLI
        _run_triposr_cli(
            inp,
            out_dir,
            model_id=model_id,
            extra_args=extra_args,
            timeout_sec=DEFAULT_TIMEOUT_S,
        )

        # Load mesh
        mesh_path = _find_mesh_file(out_dir)
        logger.info("Loading mesh from %s", mesh_path)

        loaded = trimesh.load(mesh_path, force="mesh", skip_materials=True)
        mesh = _as_trimesh(loaded)
        mesh = _normalize_to_unit_cube(mesh)
        return mesh

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            logger.info("Keeping temp dir for debug: %s", tmp_dir)
