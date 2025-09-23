from omegaconf import OmegaConf
from fastapi import Depends
from functools import lru_cache
import os

def get_config_path():
    # Choose config by ENV (e.g., small GPU vs big GPU)
    return os.getenv("GEN_CONFIG", "configs/default.yaml")

@lru_cache(maxsize=1)
def _load_config():
    cfg_path = get_config_path()
    return OmegaConf.load(cfg_path)

def get_config():
    return _load_config()

# Lazy singleton model bundle
class ModelBundle:
    def __init__(self, cfg):
        from .models.text2img_flux import load_text2img
        from .models.bg.briai_rmbg import load_rmbg
        from .models.trid.triposr import load_triposr
        from .models.validate.clip import load_clip

        self.t2i = load_text2img(cfg)
        self.rmbg = load_rmbg(cfg)
        self.triposr = load_triposr(cfg)
        self.clip = load_clip(cfg)

@lru_cache(maxsize=1)
def _load_models():
    cfg = _load_config()
    return ModelBundle(cfg)

def get_models(bundle: ModelBundle = Depends(_load_models)):
    return bundle
