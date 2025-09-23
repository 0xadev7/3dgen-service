import logging

def setup_logging(app=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    if app:
        app.logger = logging.getLogger("3dgen")
