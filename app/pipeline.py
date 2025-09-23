from __future__ import annotations
from .config import GenConfig
from .t2i import text2image, fake_image_for_debug
from .bgremove import remove_bg
from .recon import image_to_mesh
from .splats import mesh_to_gaussians
from .validate import clip_score
from .render import render_mesh_preview
import time, io
import PIL.Image as Image

class GaussianProcessor:
    def __init__(self, cfg: GenConfig, prompt: str):
        self.cfg = cfg
        self.prompt = prompt
        self.mesh = None
        self.preview = None
        self.ply_bytes = None

    def train(self, models, iters: int = 1):  # keep signature similar
        t0 = time.time()
        # 1) text-to-image
        if self.cfg.fast_debug:
            img = fake_image_for_debug(self.prompt, self.cfg)
        else:
            img = text2image(models["pipe"], self.prompt, self.cfg, seed=self.cfg.seed)

        # 2) bg removal
        cut = remove_bg(models["rmbg"], img, self.cfg)

        # 3) reconstruct
        mesh = image_to_mesh(None, cut)

        # 4) to gaussians
        ply = mesh_to_gaussians(mesh, self.cfg.splat_samples, self.cfg.splat_opacity)

        # 5) validate (using preview rendered from mesh for CLIP scoring)
        preview = render_mesh_preview(mesh, 448, 448)
        score = clip_score(models["clip_model"], models["clip_proc"], self.prompt, preview)

        # retry once if below threshold
        tries = 0
        while score < self.cfg.clip_threshold and tries < self.cfg.max_retries and not self.cfg.fast_debug:
            img = text2image(models["pipe"], self.prompt, self.cfg, seed=self.cfg.seed + tries + 1)
            cut = remove_bg(models["rmbg"], img, self.cfg)
            mesh = image_to_mesh(None, cut)
            ply = mesh_to_gaussians(mesh, self.cfg.splat_samples, self.cfg.splat_opacity)
            preview = render_mesh_preview(mesh, 448, 448)
            score = clip_score(models["clip_model"], models["clip_proc"], self.prompt, preview)
            tries += 1

        # store
        self.mesh = mesh
        self.preview = preview
        self.ply_bytes = ply

    def get_gs_model(self):
        # shim to look like the sample interface
        class _Shim:
            def __init__(self, ply_bytes: bytes):
                self._ply = ply_bytes
            def save_ply(self, buffer):
                buffer.write(self._ply)
        return _Shim(self.ply_bytes)

    def get_preview_png(self) -> bytes:
        out = io.BytesIO()
        self.preview.save(out, format="PNG")
        return out.getvalue()
