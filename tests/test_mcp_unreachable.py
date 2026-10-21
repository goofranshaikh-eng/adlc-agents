from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_mcp_unreachable(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:9999/sse")

    client = TestClient(app)

    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "type": "tool_request",
            "tool_name": "list_sales_order",
            "tool_args": {},
        }

        response = client.post("/query", json={"question": "list sales orders"})

    assert response.status_code == 200
    assert response.json() == {"error": "MCP server unreachable"}
    assert mock_llm.call_count == 1
