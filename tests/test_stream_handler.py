from __future__ import annotations

import pytest

from src.proxy.state import StateManager


def _handler_for(mapping: dict) -> "StreamHandler":
    """Build a StreamHandler whose state is pre-populated with *mapping*."""
    from src.proxy.stream_handler import StreamHandler

    sm = StateManager()
    request_id = "test-req"
    for token, original in mapping.items():
        sm.set_mapping(request_id, token, original)

    # Patch the global state_manager used by StreamHandler
    import src.proxy.stream_handler as sh_module
    original_sm = sh_module.state_manager
    sh_module.state_manager = sm
    handler = StreamHandler(request_id)
    sh_module.state_manager = original_sm
    return handler


# --------------------------------------------------------------------------- #
# _replace_tokens                                                              #
# --------------------------------------------------------------------------- #

class TestReplaceTokens:
    def test_simple_replacement(self):
        h = _handler_for({"[PERSON_1]": "Alice"})
        assert h._replace_tokens("Hello [PERSON_1]!") == "Hello Alice!"

    def test_no_mapping_returns_unchanged(self):
        h = _handler_for({})
        text = "Hello [PERSON_1]!"
        assert h._replace_tokens(text) == text

    def test_multiple_tokens(self):
        h = _handler_for({"[PERSON_1]": "Bob", "[EMAIL_ADDRESS_1]": "bob@example.com"})
        result = h._replace_tokens("Contact [PERSON_1] at [EMAIL_ADDRESS_1].")
        assert result == "Contact Bob at bob@example.com."

    def test_markdown_escaped_token(self):
        """LLMs like Gemini often emit \\[PHONE\\_NUMBER\\_1\\] in their output."""
        h = _handler_for({"[PHONE_NUMBER_1]": "+1-555-0100"})
        escaped = r"\[PHONE\_NUMBER\_1\]"
        assert h._replace_tokens(escaped) == "+1-555-0100"

    def test_json_double_escaped_token(self):
        """Inside a raw JSON stream the escaping doubles: \\\\[PHONE\\\\_NUMBER\\\\_1\\\\]."""
        h = _handler_for({"[PHONE_NUMBER_1]": "+1-555-0100"})
        # Two levels of backslash escaping
        double_escaped = r"\\[PHONE\\_NUMBER\\_1\\]"
        assert h._replace_tokens(double_escaped) == "+1-555-0100"

    def test_replacement_is_json_safe(self):
        """Originals that contain quotes/newlines must not break JSON structure."""
        h = _handler_for({"[PERSON_1]": 'John "The Boss" Doe'})
        # The safe replacement should escape the inner quotes
        result = h._replace_tokens('"name": "[PERSON_1]"')
        assert "John" in result
        # The result must still be valid JSON-like (no unescaped raw quotes inside)
        assert result.count('"') >= 2


# --------------------------------------------------------------------------- #
# _process_buffer                                                              #
# --------------------------------------------------------------------------- #

class TestProcessBuffer:
    def test_no_brackets_flushes_all(self):
        h = _handler_for({"[PERSON_1]": "Alice"})
        out, remaining = h._process_buffer("Hello world")
        assert out == "Hello world"
        assert remaining == ""

    def test_complete_token_flushes_all(self):
        h = _handler_for({"[PERSON_1]": "Alice"})
        out, remaining = h._process_buffer("Say hi to [PERSON_1] please")
        assert "Alice" in out
        assert remaining == ""

    def test_partial_token_held_in_buffer(self):
        """If a token is split mid-stream we must wait for the closing bracket."""
        h = _handler_for({"[PERSON_1]": "Alice"})
        # The token starts but has no closing ']'
        out, remaining = h._process_buffer("Hello [PERS")
        assert out == "Hello "
        assert remaining == "[PERS"

    def test_literal_bracket_json_array_flushed(self):
        """A JSON array-start '[' should not block the buffer."""
        h = _handler_for({"[PERSON_1]": "Alice"})
        out, remaining = h._process_buffer('{"choices": [{"text": "hi"}]}')
        assert remaining == ""
        assert "hi" in out

    def test_very_long_partial_token_flushed(self):
        """If the unclosed '[' prefix exceeds 50 chars it is treated as literal."""
        h = _handler_for({"[PERSON_1]": "Alice"})
        long_content = "[" + "A" * 60  # 61 chars, no closing bracket
        out, remaining = h._process_buffer(long_content)
        assert remaining == ""


# --------------------------------------------------------------------------- #
# handle_stream (async)                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_handle_stream_replaces_tokens():
    from src.proxy.stream_handler import StreamHandler
    import src.proxy.stream_handler as sh_module

    sm = StateManager()
    request_id = "stream-req"
    sm.set_mapping(request_id, "[PERSON_1]", "Alice")

    original_sm = sh_module.state_manager
    sh_module.state_manager = sm
    handler = StreamHandler(request_id)
    sh_module.state_manager = original_sm

    async def fake_stream():
        yield b"Hello "
        yield b"[PERSON_1]"
        yield b"!"

    chunks = []
    async for chunk in handler.handle_stream(fake_stream()):
        chunks.append(chunk)

    result = "".join(chunks)
    assert "Alice" in result
    assert "[PERSON_1]" not in result


@pytest.mark.asyncio
async def test_handle_stream_cleans_up_state():
    """State for the request must be deleted once the stream ends."""
    from src.proxy.stream_handler import StreamHandler
    import src.proxy.stream_handler as sh_module

    sm = StateManager()
    request_id = "cleanup-req"
    sm.set_mapping(request_id, "[PERSON_1]", "Bob")

    original_sm = sh_module.state_manager
    sh_module.state_manager = sm
    handler = StreamHandler(request_id)

    async def fake_stream():
        yield b"[PERSON_1]"

    async for _ in handler.handle_stream(fake_stream()):
        pass

    sh_module.state_manager = original_sm
    assert sm.get_mapping(request_id) == {}
