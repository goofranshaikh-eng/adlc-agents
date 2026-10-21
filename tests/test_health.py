import pytest
from fastapi.testclient import TestClient

import agent.health as health
from agent.main import app


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    health._probe_cache["reachable"] = False
    health._probe_cache["timestamp"] = None
    yield
    health._probe_cache["reachable"] = False
    health._probe_cache["timestamp"] = None


def test_health_mcp_url_unset(monkeypatch):
    monkeypatch.delenv("MCP_URL", raising=False)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up", "mcp_reachable": False}


def test_health_mcp_unreachable(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:9999/sse")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up", "mcp_reachable": False}
