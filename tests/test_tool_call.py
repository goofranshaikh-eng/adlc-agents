from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app
from tests.fastmcp_fake import assert_fastmcp_tool_call, fake_fastmcp, tool_success


def test_tool_call_success(monkeypatch):
    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
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
                    "answer": "Here are the sales orders: SO-001 ($100)",
                },
            ]

            response = client.post("/query", json={"question": "list sales orders"})

        assert response.status_code == 200
        assert "sales orders" in response.json()["answer"].lower()
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == "list_sales_order"
        assert call["params"]["arguments"] == {}


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected_args"),
    [
        ("list_sales_order", {}, {}),
        ("list_employees", {}, {}),
        ("get_employee_by_id", {"id": "emp-001"}, {"id": "EMP-001"}),
        ("get_sales_order_by_id", {"id": "so-123"}, {"id": "SO-123"}),
    ],
)
def test_four_tools_use_fastmcp_sse(monkeypatch, tool_name, tool_args, expected_args):
    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                {
                    "type": "tool_request",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
                {"type": "direct_answer", "answer": f"called {tool_name}"},
            ]
            response = client.post("/query", json={"question": tool_name})

        assert response.status_code == 200
        assert tool_name in response.json()["answer"]
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == tool_name
        assert call["params"]["arguments"] == expected_args
