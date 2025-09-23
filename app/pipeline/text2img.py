import torch, random

def generate_image(t2i_model, prompt, opt, new_seed=False):
    if new_seed:
        seed = random.randint(0, 2**31-1)
    else:
        seed = opt.text2img.seed or 12345
    g = torch.Generator(device=opt.device).manual_seed(seed)

    img = t2i_model.generate(
        prompt=prompt,
        width=opt.text2img.width,
        height=opt.text2img.height,
        steps=opt.text2img.steps,
        guidance=opt.text2img.guidance,
        generator=g,
        precision=opt.precision
    )
    return img  # PIL.Image
