from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_response(status_code: int = 200, body: bytes = b'{"ok": true}'):
    """Build a fake httpx response for proxy pass-through tests."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()

    async def _aiter_bytes():
        yield body

    resp.aiter_bytes = _aiter_bytes
    return resp


# --------------------------------------------------------------------------- #
# Health endpoints                                                             #
# --------------------------------------------------------------------------- #

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}

    def test_health_requires_no_auth(self, client):
        """Health check must be accessible without an auth token."""
        resp = client.get("/health", headers={})
        assert resp.status_code == 200

    def test_gpu_health_requires_auth(self, client):
        """/health/gpu is not in the auth bypass list and must require a token."""
        resp = client.get("/health/gpu")
        assert resp.status_code == 401

    def test_gpu_health_with_auth_returns_valid_structure(self, auth_client):
        resp = auth_client.get("/health/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "gpu" in data
        assert data["status"] in ("healthy", "degraded")


# --------------------------------------------------------------------------- #
# Auth middleware                                                              #
# --------------------------------------------------------------------------- #

class TestAuthMiddleware:
    def test_missing_token_returns_401(self, client):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_correct_bearer_token_passes_auth(self, client):
        """A valid Bearer token must pass the auth middleware (upstream mocked)."""
        mock_resp = _make_mock_response(200)
        with patch("src.api.routes.ProxyClient.get_client") as mock_client_factory:
            http = MagicMock()
            http.build_request.return_value = MagicMock()
            http.send = AsyncMock(return_value=mock_resp)
            mock_client_factory.return_value = http

            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code != 401

    def test_anthropic_x_api_key_passes_auth(self, client):
        mock_resp = _make_mock_response(200)
        with patch("src.api.routes.ProxyClient.get_client") as mock_client_factory:
            http = MagicMock()
            http.build_request.return_value = MagicMock()
            http.send = AsyncMock(return_value=mock_resp)
            mock_client_factory.return_value = http

            resp = client.post(
                "/v1/messages",
                json={"model": "claude-3-haiku", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "test-token"},
            )
        assert resp.status_code != 401

    def test_gemini_x_goog_api_key_passes_auth(self, client):
        mock_resp = _make_mock_response(200)
        with patch("src.api.routes.ProxyClient.get_client") as mock_client_factory:
            http = MagicMock()
            http.build_request.return_value = MagicMock()
            http.send = AsyncMock(return_value=mock_resp)
            mock_client_factory.return_value = http

            resp = client.post(
                "/v1beta/models/gemini-pro:generateContent",
                json={"contents": [{"parts": [{"text": "hi"}]}]},
                headers={"x-goog-api-key": "test-token"},
            )
        assert resp.status_code != 401

    def test_health_endpoint_bypasses_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_endpoint_bypasses_auth(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# RequestID middleware                                                         #
# --------------------------------------------------------------------------- #

class TestRequestIDMiddleware:
    def test_response_contains_x_request_id(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_provided_request_id_is_echoed(self, client):
        custom_id = "my-trace-id-123"
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers.get("x-request-id") == custom_id

    def test_missing_request_id_is_generated(self, client):
        resp = client.get("/health")
        rid = resp.headers.get("x-request-id", "")
        assert len(rid) > 0
