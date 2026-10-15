import pytest
from unittest.mock import Mock
from agent.src.agent_core import AgentCore
from agent.src.mcp_client import MCPClient


def test_verify_list_sales_order():
    """Test agent can invoke list_sales_order and return useful answer"""
    # Arrange
    mock_client = Mock(spec=MCPClient)
    mock_client.invoke_list_sales_order.return_value = {
        "data": [
            {"id": "SO-1", "amount": 1000},
            {"id": "SO-2", "amount": 2000},
            {"id": "SO-3", "amount": 3000}
        ]
    }

    agent = AgentCore(mcp_client=mock_client)

    # Act
    response = agent.process_query("list sales orders")

    # Assert
    assert "Found 3 sales orders" in response
    assert "SO-1" in response
    assert not response.startswith("Error")
    mock_client.invoke_list_sales_order.assert_called_once()


def test_verify_list_employees():
    """Test agent can invoke list_employees and return useful answer"""
    # Arrange
    mock_client = Mock(spec=MCPClient)
    mock_client.invoke_list_employees.return_value = {
        "data": [
            {"id": "E-1", "name": "Alice", "department": "Engineering"},
            {"id": "E-2", "name": "Bob", "department": "Sales"}
        ]
    }

    agent = AgentCore(mcp_client=mock_client)

    # Act
    response = agent.process_query("list employees")

    # Assert
    assert "Found 2 employees" in response
    assert "Alice" in response
    assert not response.startswith("Error")
    mock_client.invoke_list_employees.assert_called_once()


def test_verify_get_sales_order_by_id():
    """Test agent can invoke get_sales_order_by_id and return useful answer"""
    # Arrange
    mock_client = Mock(spec=MCPClient)
    mock_client.invoke_get_sales_order_by_id.return_value = {
        "data": {"id": "SO-1", "amount": 1000, "status": "completed"}
    }

    agent = AgentCore(mcp_client=mock_client)

    # Act
    response = agent.process_query("get sales order SO-1")

    # Assert
    assert "Sales Order SO-1" in response
    assert "Amount: 1000" in response
    assert not response.startswith("Error")
    mock_client.invoke_get_sales_order_by_id.assert_called_once_with("SO-1")


def test_verify_get_employee_by_id():
    """Test agent can invoke get_employee_by_id and return useful answer"""
    # Arrange
    mock_client = Mock(spec=MCPClient)
    mock_client.invoke_get_employee_by_id.return_value = {
        "data": {"id": "E-1", "name": "Alice", "department": "Engineering"}
    }

    agent = AgentCore(mcp_client=mock_client)

    # Act
    response = agent.process_query("get employee E-1")

    # Assert
    assert "Employee Alice" in response
    assert "ID: E-1" in response
    assert "Department: Engineering" in response
    assert not response.startswith("Error")
    mock_client.invoke_get_employee_by_id.assert_called_once_with("E-1")
