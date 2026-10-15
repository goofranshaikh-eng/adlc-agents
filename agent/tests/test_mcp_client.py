import pytest
from unittest.mock import Mock, patch
from agent.src.mcp_client import MCPClient


@patch('agent.src.mcp_client.requests.post')
def test_invoke_list_sales_order_success(mock_post):
    """Test successful invocation of list_sales_order tool"""
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "data": [
                {"id": "SO-1", "amount": 1000},
                {"id": "SO-2", "amount": 2000}
            ]
        },
        "id": 1
    }
    mock_post.return_value = mock_response

    client = MCPClient(server_url="http://test-server")

    # Act
    result = client.invoke_list_sales_order()

    # Assert
    assert result is not None
    assert "data" in result
    assert len(result["data"]) == 2

    # Verify request payload
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["params"]["name"] == "list_sales_order"
    assert payload["params"]["arguments"] == {}


@patch('agent.src.mcp_client.requests.post')
def test_invoke_list_employees_success(mock_post):
    """Test successful invocation of list_employees tool"""
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "data": [
                {"id": "E-1", "name": "Alice", "department": "Engineering"},
                {"id": "E-2", "name": "Bob", "department": "Sales"}
            ]
        },
        "id": 1
    }
    mock_post.return_value = mock_response

    client = MCPClient(server_url="http://test-server")

    # Act
    result = client.invoke_list_employees()

    # Assert
    assert result is not None
    assert "data" in result
    assert len(result["data"]) == 2

    # Verify request payload
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["params"]["name"] == "list_employees"
    assert payload["params"]["arguments"] == {}


@patch('agent.src.mcp_client.requests.post')
def test_invoke_get_sales_order_by_id_success(mock_post):
    """Test successful invocation of get_sales_order_by_id tool"""
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "data": {"id": "SO-1", "amount": 1000, "status": "completed"}
        },
        "id": 1
    }
    mock_post.return_value = mock_response

    client = MCPClient(server_url="http://test-server")

    # Act
    result = client.invoke_get_sales_order_by_id("SO-1")

    # Assert
    assert result is not None
    assert "data" in result
    assert result["data"]["id"] == "SO-1"

    # Verify request payload
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["params"]["name"] == "get_sales_order_by_id"
    assert payload["params"]["arguments"] == {"id": "SO-1"}


@patch('agent.src.mcp_client.requests.post')
def test_invoke_get_employee_by_id_success(mock_post):
    """Test successful invocation of get_employee_by_id tool"""
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "data": {"id": "E-1", "name": "Alice", "department": "Engineering"}
        },
        "id": 1
    }
    mock_post.return_value = mock_response

    client = MCPClient(server_url="http://test-server")

    # Act
    result = client.invoke_get_employee_by_id("E-1")

    # Assert
    assert result is not None
    assert "data" in result
    assert result["data"]["id"] == "E-1"

    # Verify request payload
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["params"]["name"] == "get_employee_by_id"
    assert payload["params"]["arguments"] == {"id": "E-1"}
