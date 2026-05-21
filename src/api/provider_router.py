from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    target_url: str
    auth_headers: dict[str, str]
    is_streaming_path: bool = False


def resolve(path: str, settings) -> ProviderConfig:
    """
    Maps an incoming proxy path to the upstream provider's URL, auth headers,
    and whether the path is inherently a streaming endpoint.

    Unknown paths fall back to OpenAI.
    """
    if "v1/chat/completions" in path:
        return ProviderConfig(
            target_url=f"https://api.openai.com/{path}",
            auth_headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        )

    if "v1/messages" in path:
        return ProviderConfig(
            target_url=f"https://api.anthropic.com/{path}",
            auth_headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )

    if "models" in path and any(
        keyword in path
        for keyword in ("generateContent", "run_content", "streamGenerateContent")
    ):
        # Some clients (e.g. Chatbox) send http://localhost:8000/v1 as the Gemini
        # base URL, producing doubled prefixes like v1/v1beta/ or v1/v1/.
        clean_path = (
            path.replace("v1/v1beta/", "v1beta/").replace("v1/v1/", "v1/")
        )
        return ProviderConfig(
            target_url=f"https://generativelanguage.googleapis.com/{clean_path}",
            auth_headers={"x-goog-api-key": settings.GEMINI_API_KEY},
            is_streaming_path="streamGenerateContent" in path,
        )

    # Fallback: treat as OpenAI-compatible
    auth_headers: dict[str, str] = {}
    if settings.OPENAI_API_KEY:
        auth_headers["Authorization"] = f"Bearer {settings.OPENAI_API_KEY}"
    return ProviderConfig(
        target_url=f"https://api.openai.com/{path}",
        auth_headers=auth_headers,
    )
