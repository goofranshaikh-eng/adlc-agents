from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app
from agent.mcp_client import _normalize_id
from tests.fastmcp_fake import assert_fastmcp_tool_call, fake_fastmcp


@pytest.mark.parametrize(
    ("tool_name", "raw_id", "normalized_id", "question", "answer"),
    [
        (
            "get_employee_by_id",
            "emp-001",
            "EMP-001",
            "get employee emp-001",
            "Employee EMP-001 is Alice",
        ),
        (
            "get_sales_order_by_id",
            "so-123",
            "SO-123",
            "get sales order so-123",
            "Sales order SO-123 totals 100",
        ),
        (
            "get_employee_by_id",
            "emp001",
            "EMP001",
            "get employee emp001",
            "Employee EMP001 is Alice",
        ),
        (
            "get_employee_by_id",
            "EMP-001",
            "EMP-001",
            "get employee EMP-001",
            "Employee EMP-001 is Alice",
        ),
        (
            "get_sales_order_by_id",
            "so123",
            "SO123",
            "get sales order so123",
            "Sales order SO123 totals 100",
        ),
    ],
)
def test_id_normalization(monkeypatch, tool_name, raw_id, normalized_id, question, answer):
    def handler(payload):
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "result": {
                "content": [{"type": "text", "text": answer}],
                "isError": False,
            },
        }

    with fake_fastmcp(handler) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                {
                    "type": "tool_request",
                    "tool_name": tool_name,
                    "tool_args": {"id": raw_id},
                },
                {"type": "direct_answer", "answer": answer},
            ]
            response = client.post("/query", json={"question": question})

        assert response.status_code == 200
        assert response.json()["answer"] == answer
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == tool_name
        assert call["params"]["arguments"]["id"] == normalized_id


@pytest.mark.parametrize(
    ("raw_id", "normalized_id"),
    [
        ("emp-001", "EMP-001"),
        ("Emp-001", "EMP-001"),
        ("EMP-001", "EMP-001"),
        ("emp001", "EMP001"),
        ("Emp001", "EMP001"),
        ("EMP001", "EMP001"),
        ("so-123", "SO-123"),
        ("SO-123", "SO-123"),
        ("so123", "SO123"),
        ("ord-001", "ORD-001"),
        ("Ord-001", "ORD-001"),
        ("eMp-001", "EMP-001"),
        ("foo-bar-123", "FOO-BAR-123"),
        ("emp_001", "EMP_001"),
    ],
)
def test_normalize_id_uppercases_stdio_patterns(raw_id, normalized_id):
    """Original stdio agent uppercased every string id, with no format regex."""
    assert _normalize_id({"id": raw_id}) == {"id": normalized_id}


def test_normalize_id_leaves_other_args_and_non_strings():
    assert _normalize_id({"id": "emp-001", "note": "keep"}) == {
        "id": "EMP-001",
        "note": "keep",
    }
    assert _normalize_id({"limit": 10}) == {"limit": 10}
    assert _normalize_id({"id": 1}) == {"id": 1}
