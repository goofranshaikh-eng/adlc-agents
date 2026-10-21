import pytest
from fastapi.testclient import TestClient

import agent.health as health
from agent.main import app
from tests.fastmcp_fake import fake_fastmcp, tool_success


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    health._probe_cache["reachable"] = False
    health._probe_cache["timestamp"] = None
    yield
    health._probe_cache["reachable"] = False
    health._probe_cache["timestamp"] = None


def test_health_mcp_up(monkeypatch):
    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "up", "mcp_reachable": True}
        assert server.gets, "health probe never opened the SSE endpoint"
        assert server.gets[0]["path"].rstrip("/") == "/sse"
        assert "text/event-stream" in (server.gets[0]["accept"] or "")
