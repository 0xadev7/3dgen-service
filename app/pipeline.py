from __future__ import annotations

import io
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import PIL.Image as Image

from .config import GenConfig
from .t2i import text2image, fake_image_for_debug
from .bgremove import remove_bg
from .recon import image_to_mesh
from .splats import mesh_to_gaussians
from .validate import clip_score
from .render import render_mesh_preview

logger = logging.getLogger(__name__)


@contextmanager
def _stage(name: str, extra: Optional[Dict[str, Any]] = None):
    """
    Context manager to time and log a processing stage.
    """
    extra = extra or {}
    t0 = time.perf_counter()
    logger.debug("stage.start", extra={"stage": name, **extra})
    try:
        yield
    except Exception as e:  # noqa: BLE001
        t1 = time.perf_counter()
        logger.exception(
            "stage.error",
            extra={"stage": name, "elapsed_s": round(t1 - t0, 3), **extra},
        )
        raise
    else:
        t1 = time.perf_counter()
        logger.debug(
            "stage.end",
            extra={"stage": name, "elapsed_s": round(t1 - t0, 3), **extra},
        )


class GaussianProcessor:
    """
    One-shot text→image→bg removal→mesh→gaussians pipeline with validation & retry.

    Logging:
      - Uses module logger (__name__).
      - Emits `stage.start`, `stage.end`, and `stage.error` debug/error logs.
      - Emits info logs for high-level flow and validation scores.

    Public API remains the same.
    """

    def __init__(self, cfg: GenConfig, prompt: str):
        self.cfg = cfg
        self.prompt = prompt
        self.mesh = None
        self.preview: Optional[Image.Image] = None
        self.ply_bytes: Optional[bytes] = None

        logger.info(
            "processor.init",
            extra={
                "prompt_len": len(prompt or ""),
                "fast_debug": bool(cfg.fast_debug),
                "seed": getattr(cfg, "seed", None),
                "clip_threshold": getattr(cfg, "clip_threshold", None),
                "max_retries": getattr(cfg, "max_retries", 0),
                "splat_samples": getattr(cfg, "splat_samples", None),
                "splat_opacity": getattr(cfg, "splat_opacity", None),
            },
        )

    def train(self, models, iters: int = 1) -> None:
        """
        Run the full pipeline once (iters kept for signature compatibility).

        Logs timing and retry flow. On success, sets:
          - self.mesh
          - self.preview
          - self.ply_bytes
        """
        t0 = time.perf_counter()
        logger.info("train.start", extra={"iters": iters})

        # -------------------------
        # 1) text-to-image
        # -------------------------
        with _stage("text2image", {"fast_debug": self.cfg.fast_debug}):
            if self.cfg.fast_debug:
                img = fake_image_for_debug(self.prompt, self.cfg)
                logger.debug("text2image.debug_image_emitted")
            else:
                img = text2image(
                    models["pipe"], self.prompt, self.cfg, seed=self.cfg.seed
                )
                logger.debug("text2image.generated")

        # -------------------------
        # 2) background removal
        # -------------------------
        with _stage("remove_bg"):
            cut = remove_bg(models["rmbg"], img, self.cfg)

        # -------------------------
        # 3) reconstruction
        # -------------------------
        with _stage("image_to_mesh"):
            try:
                mesh = image_to_mesh(
                    None,
                    cut,
                    model_id=getattr(self.cfg, "tripo_model_id", None),
                    extra_args=getattr(self.cfg, "tripo_extra_args", None),
                )
            except Exception:
                # Optional fallback (kept tiny + explicit)
                logger.exception("image_to_mesh failed")
                if getattr(self.cfg, "tripo_allow_fallback", True):
                    import trimesh

                    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
                else:
                    raise

        # -------------------------
        # 4) gaussians
        # -------------------------
        with _stage("mesh_to_gaussians"):
            ply = mesh_to_gaussians(
                mesh, self.cfg.splat_samples, self.cfg.splat_opacity
            )

        # -------------------------
        # 5) preview & validation
        # -------------------------
        with _stage("render_mesh_preview", {"w": 448, "h": 448}):
            preview = render_mesh_preview(mesh, 448, 448)

        with _stage("clip_score"):
            score = clip_score(
                models["clip_model"], models["clip_proc"], self.prompt, preview
            )
            logger.info("validate.score", extra={"clip_score": float(score)})

        # -------------------------
        # Retry loop (once or up to cfg.max_retries)
        # -------------------------
        tries = 0
        while (
            not self.cfg.fast_debug
            and score < self.cfg.clip_threshold
            and tries < self.cfg.max_retries
        ):
            tries += 1
            new_seed = (self.cfg.seed or 0) + tries
            logger.info(
                "retry.start",
                extra={
                    "try_idx": tries,
                    "max_retries": self.cfg.max_retries,
                    "prev_score": float(score),
                    "clip_threshold": float(self.cfg.clip_threshold),
                    "seed": new_seed,
                },
            )

            with _stage("text2image.retry", {"try_idx": tries, "seed": new_seed}):
                img = text2image(models["pipe"], self.prompt, self.cfg, seed=new_seed)

            with _stage("remove_bg.retry", {"try_idx": tries}):
                cut = remove_bg(models["rmbg"], img, self.cfg)

            with _stage("image_to_mesh.retry", {"try_idx": tries}):
                mesh = image_to_mesh(None, cut)

            with _stage("mesh_to_gaussians.retry", {"try_idx": tries}):
                ply = mesh_to_gaussians(
                    mesh, self.cfg.splat_samples, self.cfg.splat_opacity
                )

            with _stage(
                "render_mesh_preview.retry", {"try_idx": tries, "w": 448, "h": 448}
            ):
                preview = render_mesh_preview(mesh, 448, 448)

            with _stage("clip_score.retry", {"try_idx": tries}):
                score = clip_score(
                    models["clip_model"], models["clip_proc"], self.prompt, preview
                )
                logger.info(
                    "retry.score", extra={"try_idx": tries, "clip_score": float(score)}
                )

        # -------------------------
        # Store artifacts
        # -------------------------
        self.mesh = mesh
        self.preview = preview
        self.ply_bytes = ply

        t1 = time.perf_counter()
        logger.info(
            "train.end",
            extra={
                "elapsed_s": round(t1 - t0, 3),
                "final_clip_score": float(score),
                "retries_used": tries,
                "passed_threshold": bool(
                    score >= self.cfg.clip_threshold or self.cfg.fast_debug
                ),
            },
        )

    def get_gs_model(self):
        """
        Shim to match existing sample interface.
        """

        class _Shim:
            def __init__(self, ply_bytes: bytes):
                self._ply = ply_bytes

            def save_ply(self, buffer):
                buffer.write(self._ply)

        if self.ply_bytes is None:
            logger.error("get_gs_model.called_without_ply")
            raise RuntimeError("PLY bytes not available. Run train() first.")

        return _Shim(self.ply_bytes)

    def get_preview_png(self) -> bytes:
        """
        Returns the rendered preview as PNG bytes.
        """
        if self.preview is None:
            logger.error("get_preview_png.called_without_preview")
            raise RuntimeError("Preview not available. Run train() first.")

        with _stage("preview.to_png"):
            out = io.BytesIO()
            self.preview.save(out, format="PNG")
            data = out.getvalue()
            logger.debug("preview.png.bytes", extra={"bytes": len(data)})
            return data
