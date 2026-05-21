from __future__ import annotations

from src.core.token_replacer import replace


class TestReplace:
    def test_simple_replacement(self):
        assert replace("Hello [PERSON_1]!", {"[PERSON_1]": "Alice"}) == "Hello Alice!"

    def test_no_mapping_returns_unchanged(self):
        text = "Hello [PERSON_1]!"
        assert replace(text, {}) == text

    def test_empty_text_returns_unchanged(self):
        assert replace("", {"[PERSON_1]": "Alice"}) == ""

    def test_multiple_tokens(self):
        result = replace(
            "Contact [PERSON_1] at [EMAIL_ADDRESS_1].",
            {"[PERSON_1]": "Bob", "[EMAIL_ADDRESS_1]": "bob@example.com"},
        )
        assert result == "Contact Bob at bob@example.com."

    def test_markdown_escaped_token(self):
        """LLMs like Gemini emit \\[PHONE\\_NUMBER\\_1\\] in their output."""
        result = replace(r"\[PHONE\_NUMBER\_1\]", {"[PHONE_NUMBER_1]": "+1-555-0100"})
        assert result == "+1-555-0100"

    def test_json_double_escaped_token(self):
        """Inside a raw JSON stream escaping doubles: \\\\[PHONE\\\\_NUMBER\\\\_1\\\\]."""
        result = replace(r"\\[PHONE\\_NUMBER\\_1\\]", {"[PHONE_NUMBER_1]": "+1-555-0100"})
        assert result == "+1-555-0100"

    def test_replacement_is_json_safe(self):
        """Originals containing quotes/newlines must not break JSON structure."""
        result = replace('"name": "[PERSON_1]"', {"[PERSON_1]": 'John "The Boss" Doe'})
        assert "John" in result
        # Inner quotes must be escaped — result must still be parseable-ish JSON
        assert result.count('"') >= 2

    def test_longer_token_replaced_before_shorter_to_avoid_partial_match(self):
        """[EMAIL_ADDRESS_10] must not be clobbered by [EMAIL_ADDRESS_1] prefix."""
        mapping = {"[EMAIL_ADDRESS_1]": "a@b.com", "[EMAIL_ADDRESS_10]": "x@y.com"}
        result = replace("see [EMAIL_ADDRESS_10]", mapping)
        assert result == "see x@y.com"
        assert "[EMAIL_ADDRESS_1]" not in result

    def test_original_with_newline_is_json_safe(self):
        result = replace("[PERSON_1]", {"[PERSON_1]": "line1\nline2"})
        assert "\\n" in result  # newline must be escaped in JSON context
