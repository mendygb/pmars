import logging


def setup_agent_logger() -> None:
    logger = logging.getLogger("agents")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)-8s %(name)s — %(message)s"))
        logger.addHandler(h)
