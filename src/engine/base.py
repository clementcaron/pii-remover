from __future__ import annotations

from abc import ABC, abstractmethod


class AnonymizerBase(ABC):
    """
    Abstract base class for all anonymization engines.
    """
    @abstractmethod
    def anonymize(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Detects PII in *text* and replaces it with synthetic tokens.
        Returns the anonymized text and the token→original mapping.
        The caller is responsible for persisting the mapping.
        """
        pass

    @abstractmethod
    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        """
        Replaces synthetic tokens in *text* with their original values
        using the provided mapping.
        """
        pass
