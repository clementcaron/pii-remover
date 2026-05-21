from __future__ import annotations

import copy

from src.api.payload_scanner import extract, normalize, TextItem


# --------------------------------------------------------------------------- #
# normalize                                                                    #
# --------------------------------------------------------------------------- #

class TestNormalize:
    def test_strips_thinking_config(self):
        payload = {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}, "temperature": 0.7}}
        normalize(payload)
        assert "thinkingConfig" not in payload["generationConfig"]
        assert payload["generationConfig"]["temperature"] == 0.7

    def test_no_generation_config_is_noop(self):
        payload = {"model": "gpt-4o", "messages": []}
        normalize(payload)
        assert payload == {"model": "gpt-4o", "messages": []}

    def test_generation_config_not_dict_is_noop(self):
        payload = {"generationConfig": "invalid"}
        normalize(payload)
        assert payload["generationConfig"] == "invalid"

    def test_no_thinking_config_key_is_noop(self):
        payload = {"generationConfig": {"temperature": 0.5}}
        normalize(payload)
        assert payload["generationConfig"] == {"temperature": 0.5}


# --------------------------------------------------------------------------- #
# extract — OpenAI chat completions                                            #
# --------------------------------------------------------------------------- #

class TestExtractOpenAI:
    def _openai_payload(self):
        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "My name is Alice."},
            ],
        }

    def test_extracts_content_fields(self):
        items = extract(self._openai_payload())
        texts = [i.value for i in items]
        assert "You are helpful." in texts
        assert "My name is Alice." in texts

    def test_items_count(self):
        assert len(extract(self._openai_payload())) == 2

    def test_reinject_mutates_payload(self):
        payload = self._openai_payload()
        items = extract(payload)
        for item in items:
            item.container[item.key] = "[REPLACED]"
        assert payload["messages"][0]["content"] == "[REPLACED]"
        assert payload["messages"][1]["content"] == "[REPLACED]"

    def test_non_text_fields_ignored(self):
        payload = {"model": "gpt-4o", "max_tokens": 100, "messages": []}
        assert extract(payload) == []


# --------------------------------------------------------------------------- #
# extract — Anthropic messages                                                 #
# --------------------------------------------------------------------------- #

class TestExtractAnthropic:
    def _anthropic_payload(self):
        return {
            "model": "claude-3-haiku",
            "messages": [
                {"role": "user", "content": "Hello, I'm Bob."},
                {"role": "assistant", "content": "Hi Bob!"},
            ],
        }

    def test_extracts_content_fields(self):
        items = extract(self._anthropic_payload())
        texts = [i.value for i in items]
        assert "Hello, I'm Bob." in texts
        assert "Hi Bob!" in texts


# --------------------------------------------------------------------------- #
# extract — Gemini generateContent                                             #
# --------------------------------------------------------------------------- #

class TestExtractGemini:
    def _gemini_payload(self):
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What is my email address?"}],
                }
            ]
        }

    def test_extracts_text_fields(self):
        items = extract(self._gemini_payload())
        texts = [i.value for i in items]
        assert "What is my email address?" in texts

    def test_reinject_mutates_parts(self):
        payload = self._gemini_payload()
        items = extract(payload)
        for item in items:
            item.container[item.key] = "[ANON]"
        assert payload["contents"][0]["parts"][0]["text"] == "[ANON]"


# --------------------------------------------------------------------------- #
# extract — role-preceded bare strings (Anthropic list format)                #
# --------------------------------------------------------------------------- #

class TestExtractRolePrecededStrings:
    def test_string_after_role_dict_is_extracted(self):
        payload = [{"role": "user"}, "Tell me about Alice."]
        items = extract(payload)
        assert len(items) == 1
        assert items[0].value == "Tell me about Alice."

    def test_string_without_role_predecessor_not_extracted(self):
        payload = ["just a string", "another string"]
        assert extract(payload) == []


# --------------------------------------------------------------------------- #
# extract — edge cases                                                         #
# --------------------------------------------------------------------------- #

class TestExtractEdgeCases:
    def test_empty_payload(self):
        assert extract({}) == []

    def test_empty_messages_list(self):
        assert extract({"messages": []}) == []

    def test_nested_content_found(self):
        payload = {"outer": {"inner": {"content": "deep text"}}}
        items = extract(payload)
        assert len(items) == 1
        assert items[0].value == "deep text"

    def test_content_not_string_ignored(self):
        """content: [...] (list form used by some Anthropic requests) must not be collected."""
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        items = extract(payload)
        # content is a list, not a string — should not be collected at content key
        content_items = [i for i in items if i.value == [{"type": "text", "text": "hi"}]]
        assert content_items == []
