import os, uuid, io, time
from .pipeline.text2img import generate_image
from .pipeline.bg_remove import cut_foreground
from .pipeline.to_3d import single_view_mesh
from .pipeline.mesh_to_gs import mesh_to_gaussians
from .pipeline.validate import validate_gs
from .pipeline.render import render_gs_preview

class GaussianProcessor:
    def __init__(self, opt, prompt: str):
        self.opt = opt
        self.prompt = prompt
        self.run_id = str(uuid.uuid4())[:8]
        self.out_dir = os.path.join(opt.io.out_dir, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.gs_ply_path = os.path.join(self.out_dir, "model.ply")

    def train(self, models, iters: int = None):
        # 1) T2I
        img = generate_image(models.t2i, self.prompt, self.opt)
        # 2) BG
        fg = cut_foreground(models.rmbg, img, self.opt)
        # 3) 3D mesh
        mesh_path = os.path.join(self.out_dir, "recon_mesh.obj")
        single_view_mesh(models.triposr, fg, mesh_path, self.opt)
        # 4) Mesh → GS (PLY)
        mesh_to_gaussians(mesh_path, self.gs_ply_path, self.opt)
        # 5) Validate (with retry)
        for attempt in range(self.opt.validate.max_attempts):
            ok = validate_gs(models.clip, self.gs_ply_path, self.prompt, self.opt)
            if ok:
                return
            # Re-roll image (diff seed) → re-do steps
            img = generate_image(models.t2i, self.prompt, self.opt, new_seed=True)
            fg = cut_foreground(models.rmbg, img, self.opt)
            single_view_mesh(models.triposr, fg, mesh_path, self.opt)
            mesh_to_gaussians(mesh_path, self.gs_ply_path, self.opt)
        # final attempt failed: leave empty PLY to signal ignore
        open(self.gs_ply_path, "wb").close()

    def get_gs_model(self):
        return GaussianModel(self.gs_ply_path)

    def preview_png(self):
        out_png = os.path.join(self.out_dir, "preview.png")
        render_gs_preview(self.gs_ply_path, out_png, self.opt)
        return out_png

class GaussianModel:
    def __init__(self, ply_path: str):
        self.ply_path = ply_path
    def save_ply(self, fobj):
        with open(self.ply_path, "rb") as r:
            fobj.write(r.read())
