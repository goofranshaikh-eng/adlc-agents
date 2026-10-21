from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_mcp_unreachable(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:9999/sse")

    client = TestClient(app)

    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
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

        response = client.post("/query", json={"question": "list sales orders"})

    assert response.status_code == 200
    assert "unreachable" in response.json()["answer"].lower()
    tool_result = mock_llm.await_args_list[1].kwargs["tool_result"]
    assert "unreachable" in tool_result["error"].lower()
