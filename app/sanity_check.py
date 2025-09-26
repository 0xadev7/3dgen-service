import os, torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
try:
    import OpenGL, pyglet, glfw

    print("OpenGL OK, pyglet:", pyglet.__version__)
except Exception as e:
    print("OpenGL stack warning:", e)
try:
    import imageio_ffmpeg

    print("Found ffmpeg binary:", imageio_ffmpeg.get_ffmpeg_exe())
except Exception as e:
    print("FFMPEG warning:", e)
print("HF_HOME:", os.environ.get("HF_HOME"))
