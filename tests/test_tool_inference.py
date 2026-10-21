import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent.llm_client import ask_llm


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
        ("list sales orders", "list_sales_order", {}),
        ("list employees", "list_employees", {}),
        ("get employee emp-001", "get_employee_by_id", {"id": "emp-001"}),
        ("get employee emp001", "get_employee_by_id", {"id": "emp001"}),
        ("get employee Emp-001", "get_employee_by_id", {"id": "Emp-001"}),
        ("get employee foo-bar-123", "get_employee_by_id", {"id": "foo-bar-123"}),
        ("get employee emp_001", "get_employee_by_id", {"id": "emp_001"}),
        ("get employee 001", "get_employee_by_id", {"id": "001"}),
        ("Give me employee with id emp-001", "get_employee_by_id", {"id": "emp-001"}),
        ("get sales order so-123", "get_sales_order_by_id", {"id": "so-123"}),
        ("get sales order so123", "get_sales_order_by_id", {"id": "so123"}),
        ("get sales order ord-001", "get_sales_order_by_id", {"id": "ord-001"}),
        ("get sales order foo-bar-123", "get_sales_order_by_id", {"id": "foo-bar-123"}),
    ],
)
def test_ask_llm_uses_model_tool_choice(question, tool_name, tool_args):
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[_tool_use(tool_name, tool_args)]
    )
    with patch("agent.llm_client._get_client", return_value=client):
        choice = asyncio.run(ask_llm(question))

    assert choice == {
        "type": "tool_request",
        "tool_name": tool_name,
        "tool_args": tool_args,
    }
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"]
    assert {tool["name"] for tool in kwargs["tools"]} == {
        "list_sales_order",
        "list_employees",
        "get_sales_order_by_id",
        "get_employee_by_id",
    }


def test_ask_llm_returns_text_when_model_skips_tools():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[_text("Hello.")])
    with patch("agent.llm_client._get_client", return_value=client):
        choice = asyncio.run(ask_llm("hello"))

    assert choice == {"type": "direct_answer", "answer": "Hello."}


@pytest.mark.parametrize(
    ("question", "tool_name", "tool_args", "model_name", "model_args"),
    [
        (
            "get employee emp-001",
            "get_employee_by_id",
            {"id": "emp-001"},
            "list_employees",
            {},
        ),
        (
            "get employee emp001",
            "get_employee_by_id",
            {"id": "emp001"},
            "list_employees",
            {},
        ),
        (
            "get employee foo-bar-123",
            "get_employee_by_id",
            {"id": "foo-bar-123"},
            "list_employees",
            {},
        ),
        (
            "get employee emp_001",
            "get_employee_by_id",
            {"id": "emp_001"},
            "list_employees",
            {},
        ),
        (
            "get employee 001",
            "get_employee_by_id",
            {"id": "001"},
            "list_employees",
            {},
        ),
        (
            "get employee Emp-001",
            "get_employee_by_id",
            {"id": "Emp-001"},
            "get_employee_by_id",
            {},
        ),
        (
            "Give me employee with id emp-001",
            "get_employee_by_id",
            {"id": "emp-001"},
            "list_employees",
            {},
        ),
        (
            "get sales order so-123",
            "get_sales_order_by_id",
            {"id": "so-123"},
            "list_sales_order",
            {},
        ),
        (
            "get sales order so123",
            "get_sales_order_by_id",
            {"id": "so123"},
            "list_sales_order",
            {},
        ),
        (
            "get sales order ord-001",
            "get_sales_order_by_id",
            {"id": "ord-001"},
            None,
            {},
        ),
    ],
)
def test_get_by_id_handlers_extract_id_when_model_drops_it(
    question, tool_name, tool_args, model_name, model_args
):
    """Entity-by-id questions keep their tool and raw id if the model misses them."""
    client = MagicMock()
    if model_name is None:
        content = [_text("I cannot look that up.")]
    else:
        content = [_tool_use(model_name, model_args)]
    client.messages.create.return_value = MagicMock(content=content)
    with patch("agent.llm_client._get_client", return_value=client):
        choice = asyncio.run(ask_llm(question))

    assert choice == {
        "type": "tool_request",
        "tool_name": tool_name,
        "tool_args": tool_args,
    }
