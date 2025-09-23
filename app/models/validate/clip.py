import torch
import open_clip

def load_clip(cfg):
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.validate.clip_model,
        pretrained="laion2b_s32b_b79k",
        device=cfg.device
    )
    tokenizer = open_clip.get_tokenizer(cfg.validate.clip_model)
    return {"model": model, "preprocess": preprocess, "tokenizer": tokenizer}
