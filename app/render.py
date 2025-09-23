from __future__ import annotations
import numpy as np, trimesh
import PIL.Image as Image

def render_mesh_preview(mesh: trimesh.Trimesh, w: int = 640, h: int = 640) -> Image.Image:
    # Fast CPU preview by pyrender+trimesh, fallback to simple shaded
    try:
        import pyrender
        scene = pyrender.Scene(bg_color=[255,255,255,0], ambient_light=[0.4,0.4,0.4,1.0])
        tm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(tm)
        cam = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
        cam_pose = np.eye(4)
        cam_pose[:3,3] = np.array([0.0, 0.0, 2.2])   # pull back camera
        scene.add(cam, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0,1.0,1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)
        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, _ = r.render(scene)
        return Image.fromarray(color, "RGB")
    except Exception:
        # Very rough fallback: orthographic vertex scatter
        verts = mesh.vertices
        verts2d = verts[:, :2]
        verts2d = (verts2d - verts2d.min(0)) / (verts2d.ptp(0)+1e-6)
        img = np.ones((h,w,3), dtype=np.uint8)*255
        pts = (verts2d * np.array([w-1,h-1])).astype(int)
        img[pts[:,1], pts[:,0]] = (200,200,200)
        return Image.fromarray(img, "RGB")
