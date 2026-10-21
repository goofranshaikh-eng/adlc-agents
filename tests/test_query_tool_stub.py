from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_query_tool_stub():
    client = TestClient(app)

    with (
        patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm,
        patch("agent.mcp_client.call_tool", new_callable=AsyncMock) as mock_mcp,
    ):
        mock_llm.side_effect = [
            {
                "type": "tool_request",
                "tool_name": "list_sales_order",
                "tool_args": {},
            },
            {
                "type": "direct_answer",
                "answer": "I'm sorry, the MCP service is not yet available.",
            },
        ]
        mock_mcp.return_value = {"error": "MCP not implemented"}

        response = client.post("/query", json={"question": "list sales orders"})

        assert response.status_code == 200
        assert "MCP" in response.json()["answer"]

        assert mock_llm.call_count == 2
        mock_mcp.assert_called_once_with("list_sales_order", {})
