# app/pipeline/utils.py
import os
import numpy as np
from plyfile import PlyData

def load_ply_points(ply_path):
    if not ply_path or not os.path.exists(ply_path):
        return np.zeros((0,3)), np.zeros((0,3),dtype=np.uint8), np.zeros((0,3,3))
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    cols = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)

    # Optional: read covariance if present
    cov_fields = ["cov_xx","cov_xy","cov_xz","cov_yy","cov_yz","cov_zz"]
    if all(f in v.dtype.names for f in cov_fields):
        cxx, cxy, cxz, cyy, cyz, czz = (v[f].astype(np.float32) for f in cov_fields)
        cov = np.stack([
            np.stack([cxx, cxy, cxz], axis=1),
            np.stack([cxy, cyy, cyz], axis=1),
            np.stack([cxz, cyz, czz], axis=1),
        ], axis=1)
    else:
        cov = np.zeros((len(pts), 3, 3), dtype=np.float32)
    return pts, cols, cov

def clip_score(model_bundle, prompt, images):
    # TODO: replace with real OpenCLIP scoring. Placeholder returns a safe mid value.
    return 0.5

def aesthetic_score(images):
    return 6.0

def nsfw_check(images):
    return False
