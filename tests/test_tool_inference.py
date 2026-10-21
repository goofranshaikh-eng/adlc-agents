import asyncio
from unittest.mock import patch

import pytest

from agent.llm_client import _infer_tool, ask_llm


@pytest.mark.parametrize(
    ("question", "tool_name", "tool_args"),
    [
        ("list sales orders", "list_sales_order", {}),
        ("list employees", "list_employees", {}),
        ("get employee emp-001", "get_employee_by_id", {"id": "emp-001"}),
        ("get sales order so-123", "get_sales_order_by_id", {"id": "so-123"}),
    ],
)
def test_infer_tool_covers_four_tools(question, tool_name, tool_args):
    assert _infer_tool(question) == (tool_name, tool_args)


def test_ask_llm_extracts_get_by_id_args():
    """Tool requests keep the id; they must not fall through to list_employees."""
    with patch("agent.llm_client._get_client", return_value=object()):
        employee = asyncio.run(ask_llm("get employee emp-001"))
        sales_order = asyncio.run(ask_llm("get sales order so-123"))

    assert employee == {
        "type": "tool_request",
        "tool_name": "get_employee_by_id",
        "tool_args": {"id": "emp-001"},
    }
    assert sales_order == {
        "type": "tool_request",
        "tool_name": "get_sales_order_by_id",
        "tool_args": {"id": "so-123"},
    }
