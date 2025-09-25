import io, os, time, base64, json, logging, random
from contextlib import contextmanager

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time()*1000),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "extra": getattr(record, "extra", {}),
        }
        return json.dumps(payload)

def get_logger(name="plysvc"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL","INFO"))
    return logger

@contextmanager
def timed(logger, label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        logger.info(f"{label} done in {dt:.1f} ms", extra={"extra": {"stage": label, "ms": dt}})

def to_b64_bytes(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def fix_seed(seed=None):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31-1)
    import torch, numpy as np
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    return seed
