from app.deps import get_config, get_models
from app.gaussian_processor import GaussianProcessor

def test_pipeline_runs():
    cfg = get_config()
    models = get_models()
    gp = GaussianProcessor(cfg, "a cube")
    gp.train(models)
    ply = gp.gs_ply_path
    assert ply.endswith(".ply")
