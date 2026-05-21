from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

# --------------------------------------------------------------------------- #
# Environment – must be set before any src.* import                           #
# --------------------------------------------------------------------------- #
os.environ.setdefault("PROXY_AUTH_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

# --------------------------------------------------------------------------- #
# GLiNER patch – applied once at session start so the module-level            #
# `NuNERAnonymizer()` call in src/api/routes.py never hits the real model.   #
# --------------------------------------------------------------------------- #
_mock_model = MagicMock()
_mock_model.inference.return_value = [[]]   # default: no entities detected
_gliner_patch = patch("gliner.GLiNER.from_pretrained", return_value=_mock_model)
_gliner_patch.start()


def pytest_sessionfinish(session, exitstatus):
    _gliner_patch.stop()


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def mock_gliner_model():
    return _mock_model


@pytest.fixture(scope="session")
def app():
    from src.main import app as _app
    return _app


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def auth_client(app):
    """TestClient that already carries the correct auth header."""
    from starlette.testclient import TestClient
    with TestClient(app, raise_server_exceptions=True) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c
