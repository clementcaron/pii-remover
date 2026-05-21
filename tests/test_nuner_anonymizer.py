from __future__ import annotations

from unittest.mock import MagicMock


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
    def test_delegates_to_token_replacer(self):
        anon = _make_anonymizer()
        assert anon.deanonymize("Hello [PERSON_1]!", {"[PERSON_1]": "Alice"}) == "Hello Alice!"

    def test_empty_text_unchanged(self):
        anon = _make_anonymizer()
        assert anon.deanonymize("", {"[PERSON_1]": "Carol"}) == ""

    def test_no_mapping_returns_text_unchanged(self):
        anon = _make_anonymizer()
        assert anon.deanonymize("No PII here", {}) == "No PII here"


# --------------------------------------------------------------------------- #
# anonymize_batch (model mocked)                                               #
# --------------------------------------------------------------------------- #

class TestAnonymizeBatch:
    def test_empty_list_returns_empty(self):
        anon = _make_anonymizer()
        texts, mapping = anon.anonymize_batch([])
        assert texts == []
        assert mapping == {}

    def test_whitespace_only_texts_skipped(self):
        anon = _make_anonymizer()
        texts, mapping = anon.anonymize_batch(["  ", "\t"])
        anon.model.inference.assert_not_called()
        assert texts == ["  ", "\t"]
        assert mapping == {}

    def test_detected_entity_replaced_by_token(self):
        model = MagicMock()
        model.inference.return_value = [
            [{"start": 11, "end": 16, "label": "person", "text": "Alice", "score": 0.95}]
        ]
        anon = _make_anonymizer(model)
        texts, mapping = anon.anonymize_batch(["My name is Alice."])
        assert "Alice" not in texts[0]
        assert "[PERSON_1]" in texts[0]
        assert mapping["[PERSON_1]"] == "Alice"

    def test_anonymize_delegates_to_batch(self):
        model = MagicMock()
        model.inference.return_value = [[]]
        anon = _make_anonymizer(model)
        text, mapping = anon.anonymize("Hello world")
        assert text == "Hello world"
        assert mapping == {}

    def test_mapping_covers_all_entities_in_batch(self):
        """All entities across all texts in a batch end up in the returned mapping."""
        model = MagicMock()
        model.inference.return_value = [
            [{"start": 0, "end": 5, "label": "person", "text": "Alice", "score": 0.9}],
            [{"start": 5, "end": 8, "label": "person", "text": "Bob", "score": 0.9}],
        ]
        anon = _make_anonymizer(model)
        _, mapping = anon.anonymize_batch(["Alice said", "Hello Bob"])
        assert mapping.get("[PERSON_1]") == "Alice"
        assert mapping.get("[PERSON_2]") == "Bob"

