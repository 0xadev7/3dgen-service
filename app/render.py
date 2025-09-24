from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import PIL.Image as Image
import trimesh


def _clamp_size(w: int, h: int, lo: int = 32, hi: int = 4096) -> Tuple[int, int]:
    w = int(max(lo, min(hi, w)))
    h = int(max(lo, min(hi, h)))
    return w, h


def _camera_distance_for_fov(bbox_diag: float, yfov: float) -> float:
    """
    For a unit-ish mesh, pick a distance that frames it within the vertical FOV.
    """
    # Put the object inside ~80% of the FOV height.
    half_h = 0.5 * bbox_diag / math.sqrt(3)  # rough: diagonal to circumscribed sphere radius
    return (half_h / math.tan(0.5 * yfov)) * 1.25 + 0.2  # margin


def _try_render_pyrender(mesh: trimesh.Trimesh, w: int, h: int) -> Image.Image:
    import pyrender

    # Scene & mesh
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.35, 0.35, 0.35, 1.0])
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pm)

    # Camera
    yfov = np.pi / 3.0
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    bbox = mesh.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    dist = _camera_distance_for_fov(diag if np.isfinite(diag) else 1.0, yfov)

    # Isometric-ish view: rotate around Y and X a bit
    def _rotz(a):  # 4x4
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]], dtype=np.float32)

    def _roty(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=np.float32)

    def _rotx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0,  0, 0],
                         [0, c, -s, 0],
                         [0, s,  c, 0],
                         [0, 0,  0, 1]], dtype=np.float32)

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose = cam_pose @ _roty(math.radians(35)) @ _rotx(math.radians(25)) @ _rotz(math.radians(10))
    cam_pose[:3, 3] = np.array([0.0, 0.0, dist], dtype=np.float32)
    scene.add(cam, pose=cam_pose)

    # Lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    r = None
    try:
        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, _ = r.render(scene)
        return Image.fromarray(color[:, :, :3], "RGB")
    finally:
        # explicit cleanup to avoid GL context leaks
        try:
            if r is not None:
                r.delete()
        except Exception:
            pass


def _render_fallback(mesh: trimesh.Trimesh, w: int, h: int) -> Image.Image:
    """
    Fast CPU fallback: orthographic projection with tiny z-buffer and lambertian shading.
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    if len(mesh.vertices) == 0:
        return Image.fromarray(img, "RGB")

    # Apply an isometric-ish rotation for nicer views
    def rotx(a):
        c, s = math.cos(a), math.sin(a)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]], dtype=np.float32)
        return R

    def roty(a):
        c, s = math.cos(a), math.sin(a)
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=np.float32)
        return R

    V = mesh.vertices.astype(np.float32, copy=False)
    V = (V @ roty(math.radians(35)).T) @ rotx(math.radians(25)).T

    # Normalize to [0,1] in XY, keep Z for depth
    xy = V[:, :2]
    z = V[:, 2]
    if not np.isfinite(xy).all():
        xy = np.nan_to_num(xy, nan=0.0)
    if not np.isfinite(z).all():
        z = np.nan_to_num(z, nan=0.0)

    # Scale to viewport
    mins = xy.min(axis=0)
    ranges = np.maximum(xy.ptp(axis=0), 1e-6)
    xy01 = (xy - mins) / ranges
    px = np.clip((xy01[:, 0] * (w - 1)).round().astype(np.int32), 0, w - 1)
    py = np.clip((xy01[:, 1] * (h - 1)).round().astype(np.int32), 0, h - 1)

    # Simple shading using vertex normals (fall back to z-based)
    try:
        vn = mesh.vertex_normals.astype(np.float32, copy=False)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    except Exception:
        vn = np.zeros_like(V)
        vn[:, 2] = 1.0

    light_dir = np.array([0.4, 0.5, 0.75], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-8
    ndotl = np.clip((vn * light_dir).sum(axis=1), 0.0, 1.0)

    # Tiny z-buffer: prefer points with larger ndotl and closer (smaller z)
    # Normalize z to [0,1]
    z01 = (z - z.min()) / (max(z.ptp(), 1e-6))
    # Score: blend lighting and depth (closer, brighter wins)
    score = (1.2 * ndotl + (1.0 - z01)).astype(np.float32)

    zbuf = np.full((h, w), -np.inf, dtype=np.float32)
    # Color: use mesh vertex colors if present, else shaded gray
    if getattr(mesh.visual, "vertex_colors", None) is not None and \
       len(mesh.visual.vertex_colors) == len(mesh.vertices):
        col = mesh.visual.vertex_colors[:, :3].astype(np.uint8)
    else:
        base = (200.0 * (0.4 + 0.6 * ndotl)).clip(80, 230).astype(np.uint8)
        col = np.stack([base, base, base], axis=1)

    # Draw points; if a duplicate pixel, keep the one with higher score
    for i in range(len(px)):
        x, y = px[i], py[i]
        s = score[i]
        if s > zbuf[y, x]:
            zbuf[y, x] = s
            img[y, x, :] = col[i]

    return Image.fromarray(img, "RGB")


def render_mesh_preview(mesh: trimesh.Trimesh, w: int = 640, h: int = 640) -> Image.Image:
    """
    Renders a quick preview of a mesh.
      - Primary: pyrender offscreen rasterization (nice lighting).
      - Fallback: CPU-only orthographic preview with simple shading and a tiny z-buffer.

    Returns:
        PIL.Image in RGB
    """
    if not isinstance(mesh, trimesh.Trimesh):
        # If a Scene slipped through, try to merge it
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
        else:
            raise TypeError(f"render_mesh_preview expects trimesh.Trimesh, got {type(mesh)!r}")

    w, h = _clamp_size(w, h)

    # Try pyrender path first
    try:
        return _try_render_pyrender(mesh, w, h)
    except Exception:
        # Silent degrade to CPU fallback
        return _render_fallback(mesh, w, h)
