"""
Light, build-safe sanity check.
- Verifies Python deps import
- Verifies CUDA is visible to PyTorch
- Does NOT touch Hugging Face or download models
"""

import sys

def main():
    # 1) Core libs import
    try:
        import torch; import PIL; import numpy
        import trimesh; import plyfile
        import diffusers; import transformers
        from tsr.system import TSR  # TripoSR package import only
    except Exception as e:
        print({"status": "fail", "stage": "imports", "error": str(e)})
        sys.exit(1)

    # 2) CUDA presence (not required to be available during build, but we check capability)
    try:
        import torch
        # If no GPU in the builder, we just print a warning and continue
        gpu_ok = bool(torch.cuda.is_available())
        print({"status": "ok", "stage": "cuda", "cuda_available": gpu_ok})
    except Exception as e:
        print({"status": "warn", "stage": "cuda_check", "error": str(e)})

    # 3) App modules import (no model init)
    try:
        import app.pipeline.utils as _u
        import app.pipeline.text2img as _t2i
        import app.pipeline.bg_remove as _bg
        import app.pipeline.singleview_3d as _sv3d
        import app.pipeline.validate as _val
    except Exception as e:
        print({"status": "fail", "stage": "app_modules", "error": str(e)})
        sys.exit(1)

    print({"status": "ok", "stage": "done"})
    sys.exit(0)

if __name__ == "__main__":
    main()
