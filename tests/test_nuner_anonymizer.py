from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.proxy.state import StateManager


def _make_anonymizer(model: MagicMock | None = None):
    """Return a NuNERAnonymizer backed by a mock GLiNER model."""
    from src.engine.nuner import NuNERAnonymizer

    if model is None:
        model = MagicMock()
        model.inference.return_value = [[]]

    anon = NuNERAnonymizer.__new__(NuNERAnonymizer)
    anon.label_mapping = {
        "person": "PERSON",
        "email": "EMAIL_ADDRESS",
        "phone number": "PHONE_NUMBER",
        "organization": "ORG",
        "location": "LOCATION",
        "date": "DATE_TIME",
    }
    anon.labels = list(anon.label_mapping.keys())
    anon.model = model
    anon.last_timing = {}
    return anon


# --------------------------------------------------------------------------- #
# _merge_entities                                                              #
# --------------------------------------------------------------------------- #

class TestMergeEntities:
    def test_empty_list(self):
        anon = _make_anonymizer()
        assert anon._merge_entities([], "any text") == []

    def test_non_overlapping_entities_unchanged(self):
        anon = _make_anonymizer()
        entities = [
            {"start": 0, "end": 5, "label": "PERSON", "text": "Alice"},
            {"start": 10, "end": 20, "label": "EMAIL_ADDRESS", "text": "a@b.com"},
        ]
        merged = anon._merge_entities(entities, "Alice     a@b.com")
        assert len(merged) == 2

    def test_overlapping_same_label_merged(self):
        anon = _make_anonymizer()
        text = "John Doe"
        entities = [
            {"start": 0, "end": 4, "label": "PERSON", "text": "John", "score": 0.9},
            {"start": 5, "end": 8, "label": "PERSON", "text": "Doe", "score": 0.8},
        ]
        merged = anon._merge_entities(entities, text)
        assert len(merged) == 1
        assert merged[0]["text"] == "John Doe"
        assert merged[0]["start"] == 0
        assert merged[0]["end"] == 8

    def test_adjacent_entities_with_gap_merged(self):
        """Entities with a 1-char gap and same label must be merged."""
        anon = _make_anonymizer()
        text = "New York"
        entities = [
            {"start": 0, "end": 3, "label": "LOCATION", "text": "New", "score": 0.8},
            {"start": 4, "end": 8, "label": "LOCATION", "text": "York", "score": 0.85},
        ]
        merged = anon._merge_entities(entities, text)
        assert len(merged) == 1
        assert merged[0]["end"] == 8

    def test_different_labels_not_merged(self):
        anon = _make_anonymizer()
        text = "Alice ACME"
        entities = [
            {"start": 0, "end": 5, "label": "PERSON", "text": "Alice", "score": 0.9},
            {"start": 6, "end": 10, "label": "ORG", "text": "ACME", "score": 0.9},
        ]
        merged = anon._merge_entities(entities, text)
        assert len(merged) == 2

    def test_single_entity_unchanged(self):
        anon = _make_anonymizer()
        entities = [{"start": 0, "end": 5, "label": "PERSON", "text": "Alice", "score": 0.9}]
        merged = anon._merge_entities(entities, "Alice")
        assert merged == entities


# --------------------------------------------------------------------------- #
# deanonymize                                                                  #
# --------------------------------------------------------------------------- #

class TestDeanonymize:
    def _anon_with_state(self, request_id: str, mapping: dict):
        """Create anonymizer with pre-populated state."""
        import src.engine.nuner as nuner_module

        sm = StateManager()
        for token, original in mapping.items():
            sm.set_mapping(request_id, token, original)

        anon = _make_anonymizer()
        original_sm = nuner_module.state_manager
        nuner_module.state_manager = sm
        return anon, nuner_module, original_sm

    def test_simple_token_replaced(self):
        anon, mod, orig_sm = self._anon_with_state(
            "r1", {"[PERSON_1]": "Alice"}
        )
        try:
            result = anon.deanonymize("Hello [PERSON_1]!", "r1")
            assert result == "Hello Alice!"
        finally:
            mod.state_manager = orig_sm

    def test_multiple_tokens_replaced(self):
        anon, mod, orig_sm = self._anon_with_state(
            "r2",
            {"[PERSON_1]": "Bob", "[EMAIL_ADDRESS_1]": "bob@example.com"},
        )
        try:
            result = anon.deanonymize(
                "Contact [PERSON_1] at [EMAIL_ADDRESS_1].", "r2"
            )
            assert "Bob" in result
            assert "bob@example.com" in result
        finally:
            mod.state_manager = orig_sm

    def test_empty_text_unchanged(self):
        anon, mod, orig_sm = self._anon_with_state("r3", {"[PERSON_1]": "Carol"})
        try:
            assert anon.deanonymize("", "r3") == ""
        finally:
            mod.state_manager = orig_sm

    def test_no_mapping_returns_text_unchanged(self):
        anon, mod, orig_sm = self._anon_with_state("r4", {})
        try:
            text = "No PII here"
            assert anon.deanonymize(text, "r4") == text
        finally:
            mod.state_manager = orig_sm

    def test_markdown_escaped_token(self):
        """LLMs may output \\[PERSON\\_1\\]; deanonymize must still resolve it."""
        anon, mod, orig_sm = self._anon_with_state("r5", {"[PERSON_1]": "Dave"})
        try:
            result = anon.deanonymize(r"\[PERSON\_1\]", "r5")
            assert result == "Dave"
        finally:
            mod.state_manager = orig_sm


# --------------------------------------------------------------------------- #
# anonymize_batch (model mocked)                                              #
# --------------------------------------------------------------------------- #

class TestAnonymizeBatch:
    def test_empty_list_returns_empty(self):
        anon = _make_anonymizer()
        assert anon.anonymize_batch([], "req") == []

    def test_whitespace_only_texts_skipped(self):
        anon = _make_anonymizer()
        result = anon.anonymize_batch(["  ", "\t"], "req")
        # Model should not have been called
        anon.model.inference.assert_not_called()
        assert result == ["  ", "\t"]

    def test_detected_entity_replaced_by_token(self):
        model = MagicMock()
        # Model returns one person entity
        model.inference.return_value = [
            [{"start": 11, "end": 16, "label": "person", "text": "Alice", "score": 0.95}]
        ]
        anon = _make_anonymizer(model)

        import src.engine.nuner as nuner_module
        sm = StateManager()
        original_sm = nuner_module.state_manager
        nuner_module.state_manager = sm
        try:
            result = anon.anonymize_batch(["My name is Alice."], "batch-req")
            assert "Alice" not in result[0]
            assert "[PERSON_1]" in result[0]
            assert sm.get_original("batch-req", "[PERSON_1]") == "Alice"
        finally:
            nuner_module.state_manager = original_sm

    def test_anonymize_delegates_to_batch(self):
        model = MagicMock()
        model.inference.return_value = [[]]
        anon = _make_anonymizer(model)

        import src.engine.nuner as nuner_module
        sm = StateManager()
        original_sm = nuner_module.state_manager
        nuner_module.state_manager = sm
        try:
            result = anon.anonymize("Hello world", "single-req")
            assert result == "Hello world"
        finally:
            nuner_module.state_manager = original_sm
