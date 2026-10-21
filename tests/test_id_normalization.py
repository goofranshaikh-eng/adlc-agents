from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from agent.main import app


def test_id_normalization(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8000/sse")

    client = TestClient(app)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "jsonrpc": "2.0",
            "result": {"id": "EMP-001", "name": "Alice"},
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
                "tool_name": "get_employee_by_id",
                "tool_args": {"id": "emp-001"},
            },
            {
                "type": "direct_answer",
                "answer": "Employee EMP-001 is Alice",
            },
        ]
        mock_http.return_value = mock_response

        response = client.post("/query", json={"question": "get employee emp-001"})

        assert response.status_code == 200
        assert "Alice" in response.json()["answer"]

        payload = mock_http.call_args[1]["json"]
        assert payload["params"]["arguments"]["id"] == "EMP-001"
