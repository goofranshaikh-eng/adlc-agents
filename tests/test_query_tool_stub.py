from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app


@pytest.mark.parametrize(
    ("question", "tool_name", "tool_args"),
    [
        ("list sales orders", "list_sales_order", {}),
        ("list employees", "list_employees", {}),
        ("get employee emp-001", "get_employee_by_id", {"id": "emp-001"}),
        ("get sales order so-123", "get_sales_order_by_id", {"id": "so-123"}),
    ],
)
def test_query_tool_routed_to_client(question, tool_name, tool_args):
    client = TestClient(app)

    with (
        patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm,
        patch("agent.mcp_client.call_tool", new_callable=AsyncMock) as mock_mcp,
    ):
        mock_llm.return_value = {
            "type": "tool_request",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
        mock_mcp.return_value = {"error": "MCP not implemented"}

        response = client.post("/query", json={"question": question})

        assert response.status_code == 200
        assert response.json() == {"error": "MCP not implemented"}

        assert mock_llm.call_count == 1
        mock_mcp.assert_called_once_with(tool_name, tool_args)
