import os
import time
import tempfile
import io
from typing import List, Tuple, Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import trimesh

from tsr.system import TSR
from app.logutil import log

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _to_trimesh_any(mesh_obj):
    """Convert common TripoSR mesh outputs into trimesh.Trimesh."""
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj

    # dict-like (vertices/faces)
    if isinstance(mesh_obj, dict):
        v = mesh_obj.get("vertices") or mesh_obj.get("verts") or mesh_obj.get("v")
        f = mesh_obj.get("faces") or mesh_obj.get("f") or mesh_obj.get("triangles")
        if v is None or f is None:
            raise ValueError("Mesh dict missing vertices/faces.")
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
        if hasattr(f, "detach"):
            f = f.detach().cpu().numpy()
        else:
            f = np.asarray(f)
        return trimesh.Trimesh(v, f, process=False)

    # object with attributes
    for va, fa in [
        ("vertices", "faces"),
        ("verts", "faces"),
        ("v", "f"),
        ("vertices", "triangles"),
    ]:
        if hasattr(mesh_obj, va) and hasattr(mesh_obj, fa):
            v = getattr(mesh_obj, va)
            f = getattr(mesh_obj, fa)
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            if hasattr(f, "detach"):
                f = f.detach().cpu().numpy()
            else:
                f = np.asarray(f)
            return trimesh.Trimesh(v, f, process=False)

    raise TypeError(f"Unsupported mesh type: {type(mesh_obj)}")


def _trimesh_from_components(verts, faces):
    """Build a trimesh.Trimesh from torch/np verts & faces, detaching to CPU as needed."""
    import numpy as _np
    import torch as _torch

    if isinstance(verts, _torch.Tensor):
        verts = verts.detach().cpu().numpy()
    else:
        verts = _np.asarray(verts)

    if isinstance(faces, _torch.Tensor):
        faces = faces.detach().cpu().numpy()
    else:
        faces = _np.asarray(faces)

    return trimesh.Trimesh(verts, faces, process=False)


def _square_pad(image: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    w, h = image.size
    side = max(w, h)
    new_im = Image.new("RGB", (side, side), bg)
    new_im.paste(image, ((side - w) // 2, (side - h) // 2))
    return new_im


def _autocrop_foreground(image: Image.Image) -> Image.Image:
    """
    Conservative foreground crop using edges. If OpenCV is unavailable,
    returns the original image.
    """
    try:
        import cv2
    except Exception:
        return image

    gray = image.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    edges = cv2.Canny(arr, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 100 or len(ys) < 100:
        return image
    x0, x1 = max(0, xs.min() - 10), min(arr.shape[1], xs.max() + 10)
    y0, y1 = max(0, ys.min() - 10), min(arr.shape[0], ys.max() + 10)
    return image.crop((x0, y0, x1, y1))


def _preprocess_for_triposr(image: Image.Image, target=512) -> Image.Image:
    im = image.convert("RGB")
    im = _autocrop_foreground(im)
    im = _square_pad(im, (255, 255, 255))
    if max(im.size) != target:
        im = im.resize((target, target), Image.LANCZOS)
    return im


def _clean_mesh(tri: trimesh.Trimesh) -> trimesh.Trimesh:
    tri.remove_unreferenced_vertices()
    tri.remove_duplicate_faces()
    tri.remove_degenerate_faces()
    try:
        trimesh.repair.fix_inversion(tri)
        trimesh.repair.fill_holes(tri)
    except Exception:
        pass
    try:
        # mild smoothing without shrinking too much
        trimesh.smoothing.filter_taubin(tri, lamb=0.5, nu=-0.53, iterations=10)
    except Exception:
        pass
    tri.rezero()
    tri.remove_infinite_values()
    tri.remove_empty_faces()
    tri.process(validate=True)
    return tri


def render_png_from_mesh(mesh: trimesh.Trimesh, out_path: str, size: int = 512) -> str:
    """
    Render a single PNG from a mesh using headless EGL via pyrender.
    Produces a neutral-light, white-background preview at the given size.
    """
    import pyrender

    # Build scene
    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35]
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    # Optional ground plane for subtle shadow and grounding
    try:
        plane = trimesh.creation.box(extents=(10, 0.01, 10))
        # place plane slightly below mesh min Y
        plane.apply_translation([0, mesh.bounds[0][1] - 0.01, 0])
        plane_node = pyrender.Mesh.from_trimesh(plane, smooth=False)
        scene.add(plane_node)
    except Exception:
        pass

    # Camera placement
    bbox = mesh.bounds  # (2, 3)
    center = (bbox[0] + bbox[1]) * 0.5
    size_bbox = bbox[1] - bbox[0]
    radius = float(np.linalg.norm(size_bbox)) + 1e-6
    cam_dist = 1.8 * radius if radius > 0 else 1.0

    # Look-at matrix
    def look_at(eye, target, up=[0, 1, 0]):
        eye = np.asarray(eye, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        z = eye - target
        z /= np.linalg.norm(z) + 1e-9
        x = np.cross(up, z)
        x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        m = np.eye(4, dtype=np.float32)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 3] = eye
        return m

    eye = center + np.array([0.0, 0.25 * cam_dist, cam_dist], dtype=np.float32)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = look_at(eye, center)
    scene.add(cam, pose=cam_pose)

    # Lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)
    side_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    side_pose = cam_pose.copy()
    side_pose[:3, 3] = center + np.array([cam_dist, cam_dist * 0.2, cam_dist * 0.2])
    scene.add(side_light, pose=side_pose)

    r = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)
    color, _ = r.render(scene)
    Image.fromarray(color).save(out_path)
    r.delete()
    return out_path


def render_png_bytes(mesh: trimesh.Trimesh, size: int = 512) -> bytes:
    """Same as render_png_from_mesh, but returns PNG bytes (no temp files)."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "preview.png")
        render_png_from_mesh(mesh, path, size=size)
        with open(path, "rb") as f:
            return f.read()


class TextTo3DPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        model_id: Optional[str] = None,
        use_sdxl: bool = False,
    ):
        t0 = time.time()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_sd = "stabilityai/sd-turbo"
        default_sdxl = "stabilityai/sdxl-turbo"
        self.model_id = model_id or (default_sdxl if use_sdxl else default_sd)
        self.is_xl = "sdxl" in self.model_id.lower()
        log.info(f"Init pipeline device={self.device} model_id={self.model_id}")

        # ---- load text->image ----
        def _find_local(model_repo: str) -> Optional[str]:
            safe = model_repo.replace("/", "--")
            base = os.path.join(HF_HOME, "hub", f"models--{safe}", "snapshots")
            if os.path.isdir(base):
                snaps = [
                    os.path.join(base, d)
                    for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d))
                ]
                if snaps:
                    snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return snaps[0]
            return None

        local_sd = os.environ.get("MODEL_SD_DIR") or _find_local(self.model_id)
        sd_dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe_cls = StableDiffusionXLPipeline if self.is_xl else StableDiffusionPipeline
        try:
            if local_sd and os.path.isdir(local_sd):
                self.sd = pipe_cls.from_pretrained(
                    local_sd, torch_dtype=sd_dtype, safety_checker=None
                )
            else:
                self.sd = pipe_cls.from_pretrained(
                    self.model_id, torch_dtype=sd_dtype, safety_checker=None
                )
        except Exception as e:
            log.error(f"SD load failed ({self.model_id}): {e}")
            self.sd = pipe_cls.from_pretrained(
                self.model_id, torch_dtype=sd_dtype, safety_checker=None
            )
        self.sd = self.sd.to(self.device)

        # Optional LoRA hot-load (env-driven defaults)
        lora_env = os.environ.get("LORA_PATHS", "")
        lora_scales_env = os.environ.get("LORA_SCALES", "")
        lora_paths = [p for p in lora_env.split(",") if p.strip()]
        lora_scales = (
            [float(x) for x in lora_scales_env.split(",") if x.strip()]
            if lora_scales_env
            else []
        )
        if lora_paths:
            self._apply_loras(self.sd, lora_paths, lora_scales)

        # ---- load CLIP (re-ranker) ----
        clip_repo = os.environ.get("CLIP_REPO", "openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(clip_repo).eval().to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_repo)

        # ---- load TripoSR ----
        triposr_repo = os.environ.get("TRIPOSR_REPO", "stabilityai/TripoSR")
        try:
            local_tripo = os.environ.get("MODEL_TRIPOSR_DIR") or _find_local(
                triposr_repo
            )
        except Exception:
            local_tripo = None
        try:
            if local_tripo and os.path.isdir(local_tripo):
                self.tsr = TSR.from_pretrained(
                    local_tripo, config_name="config.yaml", weight_name="model.ckpt"
                )
            else:
                self.tsr = TSR.from_pretrained(
                    triposr_repo, config_name="config.yaml", weight_name="model.ckpt"
                )
            log.info("TripoSR loaded")
        except Exception as e:
            log.error(f"TripoSR load failed: {e}")
            self.tsr = TSR.from_pretrained(
                triposr_repo, config_name="config.yaml", weight_name="model.ckpt"
            )

        self.tsr.to(self.device)
        log.info(f"Init done in {time.time()-t0:.2f}s")

    # ----- helpers -----
    def _apply_loras(self, pipe, lora_paths: List[str], lora_scales: List[float]):
        if not lora_paths:
            return pipe
        for i, p in enumerate(lora_paths):
            try:
                scale = lora_scales[i] if i < len(lora_scales) else 0.8
                pipe.load_lora_weights(p)
                pipe.fuse_lora(lora_scale=scale)
                log.info(f"Applied LoRA: {p} scale={scale}")
            except Exception as e:
                log.error(f"LoRA load failed for {p}: {e}")
        return pipe

    # ----- stages -----
    @torch.inference_mode()
    def text_to_image(
        self, prompt: str, seed: int = 0, steps: int = 2, image_size: int = 512
    ) -> Image.Image:
        t0 = time.time()
        g = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        self.sd.set_progress_bar_config(disable=True)
        img = self.sd(
            prompt,
            num_inference_steps=max(1, steps),
            generator=g,
            height=image_size,
            width=image_size,
        ).images[0]
        log.info(
            f"text_to_image steps={steps} size={image_size} took {time.time()-t0:.2f}s"
        )
        return img

    @torch.inference_mode()
    def multiview_images(
        self, prompt: str, seed: int = 0, steps: int = 1, image_size: int = 384
    ):
        t0 = time.time()
        views = [
            ("front view", 0),
            ("back view", 1),
            ("left side view", 2),
            ("right side view", 3),
            ("three-quarter view", 4),
        ]
        images: List[Tuple[str, Image.Image]] = []
        for suffix, idx in views:
            g = (
                torch.Generator(device=self.device).manual_seed(seed + idx)
                if seed is not None
                else None
            )
            p = f"{prompt}, {suffix}"
            self.sd.set_progress_bar_config(disable=True)
            im = self.sd(
                p,
                num_inference_steps=max(1, steps),
                generator=g,
                height=image_size,
                width=image_size,
            ).images[0]
            images.append((suffix, im))
        log.info(f"multiview {len(images)} views in {time.time()-t0:.2f}s")
        return images

    @torch.inference_mode()
    def rank_views_with_clip(self, prompt: str, images, topk: int = 2):
        # images: list of (suffix, PIL.Image)
        texts = [prompt for _ in images]
        inputs = self.clip_processor(
            text=texts,
            images=[im for _, im in images],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            dim=-1, keepdim=True
        )
        sims = (text_embeds * image_embeds).sum(dim=-1)
        scores = sims.detach().float().cpu().tolist()
        ranked = sorted(zip(images, scores), key=lambda x: x[1], reverse=True)
        ranked = [(tag_im[0], tag_im[1], sc) for (tag_im, sc) in ranked]
        return ranked[: max(1, topk)]

    @torch.inference_mode()
    def image_to_mesh(
        self, img: Image.Image, mc_res: int = 256, vertex_color: bool = True
    ):
        """
        Robust TripoSR path with preprocessing and cleanup:
          1) Preprocess image (autocrop, square pad, resize)
          2) HWC float32 in [0,1]
          3) scene_codes = TSR([image], device)
          4) meshes = TSR.extract_mesh(scene_codes, vertex_color, resolution=mc_res)
          5) Export to PLY, reload as trimesh, clean & sanity checks
        """
        # 1) Preprocess
        img_pp = _preprocess_for_triposr(img, target=512)

        # 2) HWC float32 in [0,1]
        arr = np.asarray(img_pp, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr)

        # 3) Scene codes (list-of-one image)
        scene_codes = self.tsr([arr], device=self.device)

        # 4) Extract mesh with tunables
        meshes = self.tsr.extract_mesh(
            scene_codes, vertex_color, resolution=int(mc_res)
        )
        if not meshes or meshes[0] is None:
            raise RuntimeError("TripoSR.extract_mesh returned no mesh.")

        # 5) Export to temp PLY and reload as trimesh, then clean
        with tempfile.TemporaryDirectory() as td:
            tmp_ply = os.path.join(td, "mesh.ply")
            meshes[0].export(tmp_ply)
            tri = trimesh.load(tmp_ply, force="mesh")
            if tri is None or not isinstance(tri, trimesh.Trimesh):
                raise RuntimeError("Failed to reload exported PLY as trimesh.Trimesh.")
            tri = _clean_mesh(tri)
            # ensure normals are available
            try:
                _ = tri.vertex_normals
            except Exception:
                tri.rezero()

        return tri, {"scene_codes": scene_codes}
