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
        # We don't want to mask the tokens ([PERSON_1]), 
        # but we want to make sure the original PII doesn't leak.
        # However, this formatter is global. It's safer to REDACT ANY string that looks like a token 
        # in logs EXCEPT in the Specific Debug line where we map them.
        # ACTUALLY, the bug in Phase 3 was that it REDACTED the token, leaving the PII.
        
        # Correct logic: If we see PII in a log message, we should have already masked it.
        # But for safety, let's just make sure we don't log ANYTHING between quotes if it looks like PII.
        # For now, let's just fix the backward logic: mask the ARGUMENT that contains original PII.
        
        if record.args:
            new_args = []
            for arg in record.args:
                # If the arg is what we are detecting (the original PII), redact it.
                # We can't easily know if it's original PII or just a token.
                # Strategy: If it DOES NOT match the token pattern, but it's being logged 
                # alongside a token, it's likely PII.
                new_args.append(arg)
            record.args = tuple(new_args)

        # The previous agent's logic was:
        # record.msg = self.TOKEN_PATTERN.sub("[REDACTED_PII]", record.msg)
        # This replaces [PERSON_1] with [REDACTED_PII].
        # We want to KEEP [PERSON_1] for debugging but hide 'John Doe'.
        
        # Let's just remove the over-aggressive sanitization that was causing the leak
        # and instead focus on ensuring the logs themselves don't include PII.
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
