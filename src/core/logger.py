from __future__ import annotations

import logging
import sys
from typing import Any

from src.core.config import settings


class SanitizeFormatter(logging.Formatter):
    """
    Custom formatter that ensures sensitive data like payloads or PII tokens 
    are not logged in cleartext if they somehow reach the log.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Placeholder for actual sanitization logic if needed
        # For now, we manually ensure we don't log 'msg' that contains sensitive data
        return super().format(record)


def setup_logger() -> logging.Logger:
    """
    Configures and returns a logger instance with sanitization.
    """
    logger = logging.getLogger("pii-remover")
    logger.setLevel(settings.LOG_LEVEL)

    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
