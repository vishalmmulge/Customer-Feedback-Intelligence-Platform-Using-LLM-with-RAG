"""Structured application logging.

Logs are written to stdout so they can be captured by container / platform
logging. The log level is controlled by the ``LOG_LEVEL`` environment
variable (default ``INFO``).
"""

import logging
import sys

from .config import settings


def setup_logging() -> logging.Logger:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("app")


logger = setup_logging()
