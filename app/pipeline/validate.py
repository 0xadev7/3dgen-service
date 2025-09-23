from .render import render_views
from .utils import clip_score, aesthetic_score, nsfw_check

def validate_gs(clip_model, ply_path, prompt, opt):
    imgs = render_views(ply_path, views=opt.validate.views)
    s = clip_score(clip_model, prompt, imgs)
    if s < opt.validate.min_clip:
        return False
    if aesthetic_score(imgs) < opt.validate.min_aesthetic:
        return False
    if opt.validate.nsfw_block and nsfw_check(imgs):
        return False
    return True
