from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from agent.main import app


def test_tool_error(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8000/sse")

    client = TestClient(app)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "jsonrpc": "2.0",
            "error": {"message": "Employee not found"},
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
                "tool_args": {"id": "EMP-999"},
            },
            {
                "type": "direct_answer",
                "answer": "I'm sorry, the employee was not found.",
            },
        ]
        mock_http.return_value = mock_response

        response = client.post("/query", json={"question": "get employee emp-999"})

        assert response.status_code == 200
        assert "not found" in response.json()["answer"].lower()
