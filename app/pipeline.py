import os
import time
from typing import List, Tuple, Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

from tsr.system import TSR
import trimesh

from app.logutil import log


HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _to_trimesh_any(mesh_obj):
    """
    Convert various mesh representations from TripoSR to trimesh.Trimesh.
    Handles dicts with ('vertices'|'verts'|'v') and ('faces'|'f'|'triangles'),
    PyTorch tensors, numpy arrays, or already-constructed trimesh objects.
    """
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj

    # dict-like
    if isinstance(mesh_obj, dict):
        v = mesh_obj.get("vertices") or mesh_obj.get("verts") or mesh_obj.get("v")
        f = mesh_obj.get("faces") or mesh_obj.get("f") or mesh_obj.get("triangles")
        if v is None or f is None:
            raise ValueError(
                "Mesh dict must contain vertices/verts/v and faces/f/triangles."
            )
        if hasattr(v, "detach"):  # torch Tensor
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
        if hasattr(f, "detach"):
            f = f.detach().cpu().numpy()
        else:
            f = np.asarray(f)
        return trimesh.Trimesh(v, f, process=False)

    # object with attributes (some branches return objects with .verts/.faces or .vertices/.faces)
    attr_pairs = [
        ("vertices", "faces"),
        ("verts", "faces"),
        ("v", "f"),
        ("vertices", "triangles"),
    ]
    for va, fa in attr_pairs:
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

    raise TypeError(f"Unsupported mesh type from TripoSR: {type(mesh_obj)}")


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
        self.clip_model = CLIPModel.from_pretrained(clip_repo).to(self.device)
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
                self.tsr = TSR.from_pretrained(local_tripo, device=self.device)
            else:
                self.tsr = TSR.from_pretrained(triposr_repo, device=self.device)
            log.info("TripoSR loaded")
        except Exception as e:
            log.error(f"TripoSR load failed: {e}")
            self.tsr = TSR.from_pretrained(triposr_repo, device=self.device)

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
        ).to(self.device)
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
    def image_to_mesh(self, img: Image.Image):
        arr = np.array(img.convert("RGB"))
        out = self.tsr(arr)  # TripoSR forward; structure varies by version
        # Try standard locations:
        mesh_raw = None
        if isinstance(out, dict) and "mesh" in out:
            mesh_raw = out["mesh"]
        else:
            # Some versions return the mesh directly
            mesh_raw = out

        mesh = _to_trimesh_any(mesh_raw)
        return mesh, out
