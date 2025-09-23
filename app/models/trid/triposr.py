# app/models/trid/triposr.py
from typing import Optional
import os
import numpy as np
from PIL import Image
import trimesh

try:
    # TripoSR pip package provides a convenient API
    # pip install tripoSR
    from tripoSR.api import TripoSR
    _HAS_TRIPOSR = True
except Exception:
    _HAS_TRIPOSR = False

from huggingface_hub import snapshot_download


class TripoSRWrapper:
    """
    Thin wrapper around TripoSR for single-image 3D reconstruction to mesh.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cuda"):
        if not _HAS_TRIPOSR:
            raise RuntimeError(
                "tripoSR package not found. Please add `tripoSR` to requirements or `pip install tripoSR`."
            )
        if model_dir is None:
            # Pull official weights to local cache
            # HF repo: stabilityai/TripoSR
            model_dir = snapshot_download(repo_id="stabilityai/TripoSR", allow_patterns=["*"])
        self.model = TripoSR.from_pretrained(model_dir, device=device)

    def infer_mesh(self, rgba_img: Image.Image) -> trimesh.Trimesh:
        """
        Args:
            rgba_img: PIL RGBA or RGB
        Returns:
            trimesh.Trimesh
        """
        if rgba_img.mode != "RGB":
            rgb = rgba_img.convert("RGB")
        else:
            rgb = rgba_img
        # TripoSR API: returns (vertices, faces, (optional) vertex_colors)
        verts, faces, vcols = self.model(rgb)  # verts: (N,3) float32, faces: (M,3) int, vcols: (N,3) uint8 or None
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if vcols is not None and len(vcols) == len(verts):
            mesh.visual.vertex_colors = vcols
        return mesh

    @staticmethod
    def simplify(mesh: trimesh.Trimesh, target_faces: int = 120_000) -> trimesh.Trimesh:
        if target_faces and mesh.faces.shape[0] > target_faces:
            try:
                return mesh.simplify_quadratic_decimation(target_faces)
            except Exception:
                # If simplification backend unavailable, return original
                return mesh
        return mesh


def load_triposr(cfg):
    device = getattr(cfg, "device", "cuda")
    return TripoSRWrapper(device=device)
