from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from agent.main import app


def test_tool_call_success(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8000/sse")

    client = TestClient(app)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "jsonrpc": "2.0",
            "result": [{"id": "SO-001", "amount": 100}],
            "id": 1,
        },
        request=httpx.Request("POST", "http://127.0.0.1:8000/sse"),
    )

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
                "answer": "Here are the sales orders: SO-001 ($100)",
            },
        ]
        mock_http.return_value = mock_response

        response = client.post("/query", json={"question": "list sales orders"})

        assert response.status_code == 200
        assert "sales orders" in response.json()["answer"].lower()

        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert call_args[0][0] == "http://127.0.0.1:8000/sse"
        payload = call_args[1]["json"]
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "list_sales_order"
