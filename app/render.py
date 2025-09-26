import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio


def spin_preview(mesh, seconds=3.0, fps=16, out_path="preview.mp4", size=512):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0, keepdims=True)  # center
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pm)

    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=np.eye(4))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(cam, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)
    frames = int(seconds * fps)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for i in range(frames):
            angle = 2 * np.pi * (i / frames)
            pose = np.eye(4)
            radius = 2.2
            pose[0, 3] = radius * np.cos(angle)
            pose[1, 3] = 0.6
            pose[2, 3] = radius * np.sin(angle)
            scene.set_pose(cam_node, pose=pose)
            color, _ = r.render(scene)
            writer.append_data(color)
    r.delete()
    return out_path


def choose_best_mesh(meshes):
    """Pick a mesh by a simple heuristic: prioritize larger surface area and balanced extents."""
    import numpy as np

    best = None
    best_score = -1e9
    for tag, m in meshes:
        try:
            area = m.area
            ex = m.extents
            compact = -np.std(ex)  # prefer balanced boxes
            score = area + 0.1 * compact
        except Exception:
            score = -1e9
        if score > best_score:
            best_score = score
            best = (tag, m)
    return best
