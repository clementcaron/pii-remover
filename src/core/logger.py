from __future__ import annotations

import logging
import sys
from typing import Any

from src.core.config import settings


import re

class SanitizeFormatter(logging.Formatter):
    """
    Custom formatter that ensures sensitive data like PII tokens 
    are not logged in cleartext if they somehow reach the log.
    """
    # Regex to match tokens like [PERSON_1] or [EMAIL_ADDRESS_2]
    TOKEN_PATTERN = re.compile(r"\[[A-Z_]+_\d+\]")

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize the message if it's a string
        if isinstance(record.msg, str):
            record.msg = self.TOKEN_PATTERN.sub("[REDACTED_PII]", record.msg)
        
        # Also handle potential sensitive data in args if they were passed separately
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self.TOKEN_PATTERN.sub("[REDACTED_PII]", arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

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
        # Use our custom SanitizeFormatter
        formatter = SanitizeFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
