import logging, os, sys, time, json


def setup_logger():
    lvl = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("threegen")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    fmt = os.environ.get("LOG_FORMAT", "json")
    if fmt == "json":

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                payload = {
                    "ts": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
                    ),
                    "level": record.levelname,
                    "msg": record.getMessage(),
                    "name": record.name,
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


log = setup_logger()
