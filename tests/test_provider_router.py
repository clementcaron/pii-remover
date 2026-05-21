from __future__ import annotations

from unittest.mock import MagicMock

from src.api.provider_router import resolve, ProviderConfig


def _settings(openai="sk-openai", anthropic="sk-anthropic", gemini="sk-gemini"):
    s = MagicMock()
    s.OPENAI_API_KEY = openai
    s.ANTHROPIC_API_KEY = anthropic
    s.GEMINI_API_KEY = gemini
    return s


class TestResolveOpenAI:
    def test_chat_completions_path(self):
        cfg = resolve("v1/chat/completions", _settings())
        assert cfg.target_url == "https://api.openai.com/v1/chat/completions"
        assert cfg.auth_headers == {"Authorization": "Bearer sk-openai"}
        assert cfg.is_streaming_path is False

    def test_nested_openai_path(self):
        cfg = resolve("v1/chat/completions/extra", _settings())
        assert "api.openai.com" in cfg.target_url


class TestResolveAnthropic:
    def test_messages_path(self):
        cfg = resolve("v1/messages", _settings())
        assert cfg.target_url == "https://api.anthropic.com/v1/messages"
        assert cfg.auth_headers["x-api-key"] == "sk-anthropic"
        assert cfg.auth_headers["anthropic-version"] == "2023-06-01"
        assert cfg.is_streaming_path is False


class TestResolveGemini:
    def test_generate_content_path(self):
        cfg = resolve("v1beta/models/gemini-pro:generateContent", _settings())
        assert "generativelanguage.googleapis.com" in cfg.target_url
        assert cfg.auth_headers == {"x-goog-api-key": "sk-gemini"}
        assert cfg.is_streaming_path is False

    def test_stream_generate_content_path(self):
        cfg = resolve("v1beta/models/gemini-pro:streamGenerateContent", _settings())
        assert cfg.is_streaming_path is True

    def test_run_content_path(self):
        cfg = resolve("v1beta/models/gemini-pro:run_content", _settings())
        assert "generativelanguage.googleapis.com" in cfg.target_url

    def test_chatbox_double_v1beta_prefix_rewritten(self):
        """Chatbox sends v1/v1beta/... — router must strip the spurious v1/ prefix."""
        cfg = resolve("v1/v1beta/models/gemini-pro:generateContent", _settings())
        assert "v1/v1beta" not in cfg.target_url
        assert "v1beta/models" in cfg.target_url

    def test_chatbox_double_v1_prefix_rewritten(self):
        cfg = resolve("v1/v1/models/gemini-pro:generateContent", _settings())
        assert "v1/v1/" not in cfg.target_url


class TestResolveFallback:
    def test_unknown_path_falls_back_to_openai(self):
        cfg = resolve("some/unknown/path", _settings())
        assert "api.openai.com" in cfg.target_url
        assert cfg.auth_headers.get("Authorization") == "Bearer sk-openai"

    def test_fallback_without_openai_key_has_no_auth_header(self):
        cfg = resolve("some/unknown/path", _settings(openai=None))
        assert "Authorization" not in cfg.auth_headers
