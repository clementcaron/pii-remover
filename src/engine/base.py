from __future__ import annotations

from abc import ABC, abstractmethod


class AnonymizerBase(ABC):
    """
    Abstract base class for all anonymization engines.
    """
    @abstractmethod
    def anonymize(self, text: str, request_id: str) -> str:
        """
        Detects PII in the given text and replaces it with tokens.
        Updates the global state manager with the mapping.
        """
        pass

    @abstractmethod
    def deanonymize(self, text: str, request_id: str) -> str:
        """
        Replaces tokens in the given text with their original PII.
        Uses the mapping from the state manager for the given request_id.
        """
        pass
