from __future__ import annotations
import numpy as np, trimesh
from typing import Tuple
import io

def sample_points_from_mesh(mesh: trimesh.Trimesh, n: int) -> Tuple[np.ndarray, np.ndarray]:
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, n)
    normals = mesh.face_normals[face_idx]
    # colors: if vertex colors exist, barycentric sample; else default gray
    if mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
        vc = mesh.visual.vertex_colors[:,:3]
        faces = mesh.faces[face_idx]
        b = trimesh.triangles.points_to_barycentric(mesh.triangles[face_idx], pts)
        colors = (vc[faces] * b[...,None]).sum(axis=1).astype(np.uint8)
    else:
        colors = np.full((n,3), 200, dtype=np.uint8)
    return pts.astype(np.float32), normals.astype(np.float32), colors

def gaussian_attributes(n: int, normals: np.ndarray, opacity: float):
    # Simple covariance aligned to normal; small anisotropy
    # Store as 3x3 upper-triangular (or 6 params) – we’ll export as simple radii & rotation-less variant
    # Many viewers accept per-point radius; baseline GaussianLab often expects full SH; we keep simple
    radii = np.full((n,), 0.01, dtype=np.float32)
    opac = np.full((n,), opacity, dtype=np.float32)
    return radii, opac

def to_ply_gaussians(points: np.ndarray, normals: np.ndarray, colors: np.ndarray, radii: np.ndarray, opacity: np.ndarray) -> bytes:
    # Export a point-based Gaussian PLY (lightweight). If your validator expects the
    # 3DGS research PLY (with SH coeffs), it will still parse points/colors/opacity; we
    # keep attributes minimal for speed & compatibility.
    n = points.shape[0]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(n)
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "property float radius\nproperty float opacity\nend_header\n"
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))
    rec = np.zeros(n, dtype=[
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('nx','<f4'),('ny','<f4'),('nz','<f4'),
        ('r','<u1'),('g','<u1'),('b','<u1'),
        ('radius','<f4'),('opacity','<f4'),
    ])
    rec['x']=points[:,0]; rec['y']=points[:,1]; rec['z']=points[:,2]
    rec['nx']=normals[:,0]; rec['ny']=normals[:,1]; rec['nz']=normals[:,2]
    rec['r']=colors[:,0]; rec['g']=colors[:,1]; rec['b']=colors[:,2]
    rec['radius']=radii; rec['opacity']=opacity
    buf.write(rec.tobytes())
    return buf.getvalue()

def mesh_to_gaussians(mesh: trimesh.Trimesh, n_samples: int, opacity: float=0.85) -> bytes:
    pts, nrm, col = sample_points_from_mesh(mesh, n_samples)
    rad, op = gaussian_attributes(n_samples, nrm, opacity)
    return to_ply_gaussians(pts, nrm, col, rad, op)
