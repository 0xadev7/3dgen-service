import os, numpy as np, torch
from PIL import Image
import trimesh
from plyfile import PlyData, PlyElement
from .utils import get_logger, timed

logger = get_logger()

class SingleView3D:
    def __init__(self, device="cuda"):
        self.device = device
        with timed(logger, "load_triposr"):
            from tsr.system import TSR
            ckpt = os.getenv("TRIPOSR_MODEL_ID", "stabilityai/TripoSR")
            self.tsr = TSR.from_pretrained(ckpt, cache_dir=os.getenv("HF_HOME", None)).to(self.device)
            logger.info("TripoSR loaded", extra={"extra":{"model_id": ckpt}})

    @torch.inference_mode()
    def to_mesh(self, rgba: Image.Image):
        with timed(logger, "triposr_infer"):
            import numpy as np
            arr = np.array(rgba)
            mesh = self.tsr(arr)  # returns a trimesh.Trimesh
        return mesh

    def mesh_to_pointcloud(self, mesh: trimesh.Trimesh, n_points=200_000):
        with timed(logger, "sample_points"):
            pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
            if mesh.visual.kind == 'texture' and hasattr(mesh.visual, "uv"):
                colors = np.full((len(pts), 3), 255, dtype=np.uint8)
            else:
                if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                    base_colors = np.array(mesh.visual.vertex_colors[:,:3])
                    colors = base_colors[face_idx % len(base_colors)]
                else:
                    colors = np.full((len(pts),3), 200, dtype=np.uint8)
        return pts, colors

    def write_ply(self, pts: np.ndarray, colors: np.ndarray, out_path: str):
        with timed(logger, "write_ply"):
            verts = np.empty(pts.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
            verts['x'], verts['y'], verts['z'] = pts[:,0], pts[:,1], pts[:,2]
            verts['red'], verts['green'], verts['blue'] = colors[:,0], colors[:,1], colors[:,2]
            PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(out_path)
        return out_path
