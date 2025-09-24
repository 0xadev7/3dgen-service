from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def _clamp01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _safe_normals(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> np.ndarray:
    """
    Return per-sample normals (N, 3). Prefers face normals; falls back to vertex normals if needed.
    """
    try:
        fn = mesh.face_normals  # (F, 3) computed lazily by trimesh
    except Exception:
        logger.debug("mesh.face_normals unavailable; synthesizing zeros.")
        fn = np.zeros((len(mesh.faces), 3), dtype=np.float32)

    normals = fn[face_idx] if len(fn) > 0 else np.zeros((len(face_idx), 3), dtype=np.float32)

    # Normalize, guard against zeros/NaNs
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(n, 1e-8), out=np.zeros_like(normals), where=(n > 1e-12))
    return normals.astype(np.float32)


def _get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """
    Returns per-vertex RGB uint8 colors if present, else None.
    """
    if getattr(mesh.visual, "kind", None) == "vertex" and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
            # vc may be RGBA; take first three channels
            return vc[:, :3].astype(np.uint8)
    return None


def _get_face_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """
    Returns per-face RGB uint8 colors if present, else None.
    """
    if getattr(mesh.visual, "kind", None) == "face" and hasattr(mesh.visual, "face_colors"):
        fc = np.asarray(mesh.visual.face_colors)
        if fc.ndim == 2 and fc.shape[0] == len(mesh.faces):
            return fc[:, :3].astype(np.uint8)
    return None


def _barycentric_vertex_color(
    mesh: trimesh.Trimesh, pts: np.ndarray, face_idx: np.ndarray, vcolors: np.ndarray
) -> np.ndarray:
    """
    Interpolate vertex colors at sampled points using barycentric coords.
    """
    # faces: (N, 3) vertex indices used by each sampled face
    faces = mesh.faces[face_idx]  # (N, 3)

    # triangles: (N, 3, 3) positions of triangle vertices
    tri = mesh.triangles[face_idx]  # (N, 3, 3)

    # barycentric coords for each point in its triangle
    b = trimesh.triangles.points_to_barycentric(tri, pts)  # (N, 3)
    # sanitize NaNs (degenerate triangles); default to uniform weights
    bad = ~np.isfinite(b).all(axis=1)
    if np.any(bad):
        b[bad] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    # gather vertex colors for each corner
    c0 = vcolors[faces[:, 0]]
    c1 = vcolors[faces[:, 1]]
    c2 = vcolors[faces[:, 2]]

    colors = (c0 * b[:, 0:1] + c1 * b[:, 1:2] + c2 * b[:, 2:3]).astype(np.uint8)
    return colors


def _compute_radius(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    """
    Pick a reasonable radius based on mesh scale.
    Assumes upstream normalized mesh (~unit box), but still compute robustly.
    """
    bbox = mesh.bounds  # (2, 3)
    diag = float(np.linalg.norm(bbox[1] - bbox[0])) if np.all(np.isfinite(bbox)) else 1.0
    base = max(diag, 1e-6) * 0.01  # ~1% of diagonal
    r = np.full((n,), base, dtype=np.float32)
    return r


# ---------------------------
# Public API (drop-in)
# ---------------------------
def sample_points_from_mesh(mesh: trimesh.Trimesh, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples N points on the mesh surface and returns:
      points:  (N, 3) float32
      normals: (N, 3) float32
      colors:  (N, 3) uint8
    Color priority: vertex colors → face colors → mid-gray.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    # Try even sampling; fall back to generic sampling if needed.
    try:
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, n)
    except Exception as e:
        logger.debug("sample_surface_even failed (%s); falling back to sample_surface.", e)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)

    pts = pts.astype(np.float32, copy=False)
    normals = _safe_normals(mesh, face_idx)

    # Colors
    vcolors = _get_vertex_colors(mesh)
    if vcolors is not None:
        colors = _barycentric_vertex_color(mesh, pts, face_idx, vcolors)
    else:
        fcolors = _get_face_colors(mesh)
        if fcolors is not None:
            colors = fcolors[face_idx]
        else:
            colors = np.full((n, 3), 200, dtype=np.uint8)

    return pts, normals, colors


def gaussian_attributes(n: int, normals: np.ndarray, opacity: float):
    """
    Returns (radii, opacity) arrays sized (N,), clamped to safe ranges.
    """
    op = np.full((n,), float(_clamp01(opacity)), dtype=np.float32)
    # Radii from mesh scale (caller passes the same mesh; we don't have it here)
    # We'll set placeholder; real radii computed in mesh_to_gaussians with mesh info.
    # (Kept for API compatibility.)
    rad = np.full((n,), 0.01, dtype=np.float32)
    return rad, op


def to_ply_gaussians(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    opacity: np.ndarray,
) -> bytes:
    """
    Write a compact binary little-endian PLY with fields:
      x y z nx ny nz red green blue radius opacity
    """
    n = int(points.shape[0])
    if not (normals.shape == (n, 3) and colors.shape == (n, 3) and radii.shape == (n,) and opacity.shape == (n,)):
        raise ValueError("Attribute shapes do not match point count.")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float radius\nproperty float opacity\n"
        "end_header\n"
    )

    rec = np.zeros(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("r", "<u1"),
            ("g", "<u1"),
            ("b", "<u1"),
            ("radius", "<f4"),
            ("opacity", "<f4"),
        ],
    )
    rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
    rec["nx"], rec["ny"], rec["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["radius"], rec["opacity"] = radii.astype(np.float32), opacity.astype(np.float32)

    buf = io.BytesIO()
    buf.write(header.encode("ascii"))
    buf.write(rec.tobytes(order="C"))
    return buf.getvalue()


def mesh_to_gaussians(mesh: trimesh.Trimesh, n_samples: int, opacity: float = 0.85) -> bytes:
    """
    End-to-end: sample from mesh, generate attributes, and export Gaussian PLY bytes.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be trimesh.Trimesh, got {type(mesh)!r}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    pts, nrm, col = sample_points_from_mesh(mesh, n_samples)

    # Radii: scale-aware (prefer mesh extents)
    bbox = mesh.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0])) if np.all(np.isfinite(bbox)) else 1.0
    base_r = max(diag, 1e-6) * 0.01
    rad = np.full((n_samples,), base_r, dtype=np.float32)

    # Opacity (clamped)
    op = np.full((n_samples,), float(_clamp01(opacity)), dtype=np.float32)

    return to_ply_gaussians(pts, nrm, col, rad, op)
