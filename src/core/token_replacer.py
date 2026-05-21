from __future__ import annotations

import json
import re


def replace(text: str, mapping: dict[str, str]) -> str:
    """
    Replaces anonymization tokens in *text* with their original values from *mapping*.

    Handles two escaping variants LLMs emit around tokens:
      - Markdown: \\[PHONE\\_NUMBER\\_1\\]
      - JSON-encoded Markdown: \\\\[PHONE\\\\_NUMBER\\\\_1\\\\]

    The replacement value is JSON-safe-escaped so that originals containing
    quotes or newlines cannot break surrounding JSON structure.
    """
    if not text or not mapping:
        return text

    for token, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        safe_original = json.dumps(original)[1:-1]

        escaped_token = ""
        for char in token:
            if char in ["_", "[", "]"]:
                # Allow 0, 1, or 2 leading backslashes — covers plain, Markdown-escaped,
                # and JSON-encoded-Markdown-escaped variants.
                escaped_token += r"\\{0,2}" + re.escape(char)
            else:
                escaped_token += re.escape(char)

        text = re.sub(escaped_token, lambda m: safe_original, text)

    return text
