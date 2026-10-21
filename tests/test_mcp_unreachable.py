from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from agent.main import app


def test_mcp_unreachable(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:9999/sse")

    client = TestClient(app)

    with (
        patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm,
        patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http,
    ):
        mock_llm.side_effect = [
            {
                "type": "tool_request",
                "tool_name": "list_sales_order",
                "tool_args": {},
            },
            {
                "type": "direct_answer",
                "answer": "I'm sorry, the MCP server is unreachable.",
            },
        ]
        mock_http.side_effect = httpx.ConnectError(
            "Connection refused",
            request=httpx.Request("POST", "http://127.0.0.1:9999/sse"),
        )

        response = client.post("/query", json={"question": "list sales orders"})

        assert response.status_code == 200
        assert "unreachable" in response.json()["answer"].lower()
