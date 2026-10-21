from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app
from agent.mcp_client import _normalize_id
from tests.fastmcp_fake import assert_fastmcp_tool_call, fake_fastmcp


def _assert_query_sends_id(monkeypatch, tool_name, raw_id, normalized_id, question, answer):
    """POST /query and require the FastMCP arguments to carry the normalized id."""

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
        (
            "get_employee_by_id",
            "foo-bar-123",
            "FOO-BAR-123",
            "get employee foo-bar-123",
            "Employee FOO-BAR-123 is Charlie",
        ),
        (
            "get_employee_by_id",
            "Foo-Bar-123",
            "FOO-BAR-123",
            "get employee Foo-Bar-123",
            "Employee FOO-BAR-123 is Charlie",
        ),
        (
            "get_employee_by_id",
            "emp_001",
            "EMP_001",
            "get employee emp_001",
            "Employee EMP_001 is Dana",
        ),
    ],
)
def test_id_normalization(monkeypatch, tool_name, raw_id, normalized_id, question, answer):
    _assert_query_sends_id(
        monkeypatch, tool_name, raw_id, normalized_id, question, answer
    )


def test_id_normalization_hyphenated_employee(monkeypatch):
    """Hyphenated employee ids are uppercased: emp-001 → EMP-001."""
    _assert_query_sends_id(
        monkeypatch,
        "get_employee_by_id",
        "emp-001",
        "EMP-001",
        "get employee emp-001",
        "Employee EMP-001 is Alice",
    )


def test_id_normalization_hyphenated_sales_order(monkeypatch):
    """Hyphenated sales-order ids are uppercased: so-123 → SO-123."""
    _assert_query_sends_id(
        monkeypatch,
        "get_sales_order_by_id",
        "so-123",
        "SO-123",
        "get sales order so-123",
        "Sales order SO-123 totals 100",
    )


def test_id_normalization_or_passthrough_non_hyphenated(monkeypatch):
    """Unhyphenated ids are uppercased, not passed through.

    The original stdio orchestrator used ``value.upper() if key == "id"``
    with no format regex, so emp001 becomes EMP001.
    """
    _assert_query_sends_id(
        monkeypatch,
        "get_employee_by_id",
        "emp001",
        "EMP001",
        "get employee emp001",
        "Employee EMP001 is Bob",
    )


def test_id_multi_part_prefix_passthrough(monkeypatch):
    """Multi-part ids are uppercased. The original rule was not passthrough.

    ``foo-bar-123`` and ``Foo-Bar-123`` both become ``FOO-BAR-123``. A
    prefix-digits regex would leave the mixed-case form unchanged.
    """
    _assert_query_sends_id(
        monkeypatch,
        "get_employee_by_id",
        "Foo-Bar-123",
        "FOO-BAR-123",
        "get employee Foo-Bar-123",
        "Employee FOO-BAR-123 is Charlie",
    )


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
        ("Foo-Bar-123", "FOO-BAR-123"),
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
