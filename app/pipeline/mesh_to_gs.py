import numpy as np
import trimesh

def mesh_to_gaussians(mesh_path, out_ply, opt):
    mesh = trimesh.load(mesh_path, force='mesh')
    # sample points on surface
    pts, face_idx = trimesh.sample.sample_surface(mesh, 
                                                  int(min(opt.gs.max_points, 
                                                          mesh.area * opt.gs.points_per_m2)))
    # normals for anisotropic covariance
    normals = mesh.face_normals[face_idx]
    # build per-point covariance as oriented ellipsoids
    # Simple heuristic: principal axis along normal.
    covs = []
    s = opt.gs.cov_scale
    for n in normals:
        # Oriented covariance: normal axis larger spread
        # Covariance matrix in local frame
        C = np.diag([s*0.5, s*0.5, s*1.0])
        # Align z-axis to normal n
        z = n / (np.linalg.norm(n) + 1e-8)
        # Construct orthonormal basis (x,y,z)
        x = np.array([1,0,0])
        if abs(np.dot(x,z)) > 0.9:
            x = np.array([0,1,0])
        y = np.cross(z, x); y/=np.linalg.norm(y)+1e-8
        x = np.cross(y, z); x/=np.linalg.norm(x)+1e-8
        R = np.vstack([x,y,z])
        cov = R.T @ C @ R
        covs.append(cov)
    covs = np.stack(covs, axis=0)

    # basic color: sample vertex colors or texture (fallback white)
    colors = np.clip(mesh.visual.to_color().vertex_colors[:len(pts), :3], 0, 255).astype(np.uint8)

    # write PLY with gaussian attributes (mean, cov as 6 unique terms for symmetric matrix)
    write_gaussian_ply(pts, colors, covs, out_ply)

def write_gaussian_ply(pts, colors, covs, out_ply):
    # pack upper-triangular covariance (xx, xy, xz, yy, yz, zz)
    cov_ut = np.stack([covs[:,0,0], covs[:,0,1], covs[:,0,2], covs[:,1,1], covs[:,1,2], covs[:,2,2]], axis=1)
    with open(out_ply, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(pts)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float cov_xx
property float cov_xy
property float cov_xz
property float cov_yy
property float cov_yz
property float cov_zz
end_header
"""
        f.write(header.encode("ascii"))
        rec = np.zeros(len(pts), dtype=[
            ('x','<f4'),('y','<f4'),('z','<f4'),
            ('r','u1'),('g','u1'),('b','u1'),
            ('cxx','<f4'),('cxy','<f4'),('cxz','<f4'),
            ('cyy','<f4'),('cyz','<f4'),('czz','<f4'),
        ])
        rec['x'], rec['y'], rec['z'] = pts[:,0], pts[:,1], pts[:,2]
        rec['r'], rec['g'], rec['b'] = colors[:,0], colors[:,1], colors[:,2]
        rec['cxx'], rec['cxy'], rec['cxz'], rec['cyy'], rec['cyz'], rec['czz'] = cov_ut.T
        rec.tofile(f)
