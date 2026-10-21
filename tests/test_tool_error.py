from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app
from tests.fastmcp_fake import fake_fastmcp


def test_tool_error(monkeypatch):
    def handler(payload):
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32000, "message": "Employee not found"},
        }

    with fake_fastmcp(handler) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "type": "tool_request",
                "tool_name": "get_employee_by_id",
                "tool_args": {"id": "EMP-999"},
            }

            response = client.post("/query", json={"question": "get employee emp-999"})

        assert response.status_code == 200
        assert response.json() == {"error": "Employee not found"}
        assert mock_llm.call_count == 1
