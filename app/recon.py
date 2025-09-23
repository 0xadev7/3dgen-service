from __future__ import annotations
import torch, numpy as np
import PIL.Image as Image
import trimesh

@torch.no_grad()
def image_to_mesh(tripo, img: Image.Image) -> trimesh.Trimesh:
    # FAST_DEBUG or skipped model: make a simple unit sphere (nice for plumbing tests)
    if tripo is None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        return mesh

    # Normal path: TripoSR -> mesh
    result = tripo.reconstruct(img)  # dict or path depending on version
    if isinstance(result, dict) and "mesh" in result:
        m = result["mesh"]
        mesh = m if isinstance(m, trimesh.Trimesh) else trimesh.load(m, force="mesh")
    else:
        mesh = trimesh.load(result, force="mesh")

    # Normalize to unit cube
    mesh = mesh.copy()
    if not mesh.is_watertight:
        mesh.remove_unreferenced_vertices()
    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2.0
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / max(scale, 1e-6))
    return mesh
