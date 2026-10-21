from unittest.mock import AsyncMock, patch

import httpx
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


def test_health_mcp_up(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8000/sse")

    client = TestClient(app)

    with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
        mock_head.return_value = httpx.Response(
            status_code=200,
            request=httpx.Request("HEAD", "http://127.0.0.1:8000/sse"),
        )

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "up", "mcp_reachable": True}
