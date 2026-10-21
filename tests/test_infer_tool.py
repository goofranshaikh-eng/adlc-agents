from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import agent.llm_client as llm_client
from agent.main import app
from tests.fastmcp_fake import assert_fastmcp_tool_call, fake_fastmcp, tool_success


def _tool_use(name: str, args: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    return block


def _text(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


@pytest.mark.parametrize(
    ("question", "tool_name", "tool_args"),
    [
        ("get employee emp-001", "get_employee_by_id", {"id": "EMP-001"}),
        ("get employee emp001", "get_employee_by_id", {"id": "EMP001"}),
        ("get employee Emp-001", "get_employee_by_id", {"id": "EMP-001"}),
        ("get sales order so-123", "get_sales_order_by_id", {"id": "SO-123"}),
        ("get sales order so123", "get_sales_order_by_id", {"id": "SO123"}),
        ("get sales order ord-001", "get_sales_order_by_id", {"id": "ORD-001"}),
        ("list employees", "list_employees", {}),
        ("list sales orders", "list_sales_order", {}),
    ],
)
def test_question_routes_over_fastmcp_sse(monkeypatch, question, tool_name, tool_args):
    """The model selects the tool; the SSE session receives the normalized call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm_client._client = None

    raw_args = {}
    if tool_name.startswith("get_"):
        raw_args = {"id": question.rsplit(None, 1)[-1]}

    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)
        with patch("agent.llm_client.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = [
                MagicMock(content=[_tool_use(tool_name, raw_args)]),
                MagicMock(content=[_text(f"answered {tool_name}")]),
            ]
            response = client.post("/query", json={"question": question})

        assert response.status_code == 200
        assert response.json()["answer"] == f"answered {tool_name}"
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == tool_name
        assert call["params"]["arguments"] == tool_args

    llm_client._client = None
