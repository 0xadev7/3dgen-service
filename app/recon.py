from __future__ import annotations
import torch, numpy as np
from typing import Tuple
import PIL.Image as Image
import trimesh

# Wrap TripoSR inference to a clean API
@torch.no_grad()
def image_to_mesh(tripo, img: Image.Image) -> trimesh.Trimesh:
    # TripoSR expects PIL image; returns mesh dict or file path depending on version
    # We standardize to trimesh.Trimesh
    result = tripo.reconstruct(img)  # returns dict with 'mesh' as Trimesh or path
    mesh: trimesh.Trimesh
    if isinstance(result, dict) and "mesh" in result:
        m = result["mesh"]
        if isinstance(m, trimesh.Trimesh):
            mesh = m
        else:
            mesh = trimesh.load(m, force="mesh")
    else:
        # fallback if API differs
        mesh = trimesh.load(result, force="mesh")
    # Normalize: center + scale to unit cube for consistent splats
    mesh = mesh.copy()
    if not mesh.is_watertight:
        mesh.remove_unreferenced_vertices()
    # Normalize
    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2.0
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / max(scale, 1e-6))
    return mesh
