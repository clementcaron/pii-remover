from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class TextItem:
    """
    A reference to a text field inside a provider payload.
    *container* is the dict or list that owns the field; *key* is the index
    or key within it; *value* is the current text.

    Calling `container[key] = new_text` re-injects anonymized text in place.
    """
    container: Union[dict, list]
    key: Union[str, int]
    value: str


def normalize(payload: dict) -> None:
    """
    Applies in-place payload normalisations before traversal.

    Currently: strips ``generationConfig.thinkingConfig`` from Gemini payloads
    to prevent 30-second stalls and hidden tokens that third-party clients
    (e.g. Chatbox) cannot render.
    """
    if isinstance(payload, dict):
        gen_cfg = payload.get("generationConfig")
        if isinstance(gen_cfg, dict):
            gen_cfg.pop("thinkingConfig", None)


def extract(payload) -> list[TextItem]:
    """
    Recursively traverses *payload* and returns a list of :class:`TextItem`
    objects for every text field that should be anonymized.

    Recognised patterns:
    - Dict key ``"content"`` or ``"text"`` whose value is a string
      (OpenAI messages, Gemini ``parts``, Anthropic content blocks).
    - A bare string element inside a list that is immediately preceded by
      a dict containing a ``"role"`` key (Anthropic / OpenAI role-content
      pairs stored as alternating list items).
    """
    items: list[TextItem] = []
    _collect(payload, items)
    return items


def _collect(data, target: list[TextItem]) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("content", "text") and isinstance(v, str):
                target.append(TextItem(container=data, key=k, value=v))
            else:
                _collect(v, target)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            predecessor = data[i - 1] if i > 0 else {}
            if (
                isinstance(item, str)
                and isinstance(predecessor, dict)
                and "role" in predecessor
            ):
                target.append(TextItem(container=data, key=i, value=item))
            else:
                _collect(item, target)
