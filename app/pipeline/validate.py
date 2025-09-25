import numpy as np
from .utils import get_logger
logger = get_logger()

def quick_validate(mask, pts, bbox_min_points=50_000):
    ok = True
    reasons = []

    cov = float((mask > 0.5).mean())
    if cov < 0.08:
        ok = False; reasons.append(f"low_foreground_coverage={cov:.3f}")

    if pts is None or len(pts) < bbox_min_points:
        ok = False; reasons.append(f"few_points={0 if pts is None else len(pts)}")

    if pts is not None:
        spread = np.std(pts, axis=0)
        if float(spread.max()) < 1e-3:
            ok = False; reasons.append("degenerate_spread")

    logger.info("validation", extra={"extra":{"ok": ok, "reasons": reasons}})
    return ok, reasons
