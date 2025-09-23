"""
Quick perf benchmark: run one pipeline end-to-end.
"""
import time
from app.deps import get_config, get_models
from app.gaussian_processor import GaussianProcessor

if __name__ == "__main__":
    cfg = get_config()
    models = get_models()
    prompt = "a small red robot"
    t0 = time.time()
    gp = GaussianProcessor(cfg, prompt)
    gp.train(models)
    t1 = time.time()
    print(f"Benchmark took {(t1-t0):.2f} sec")
