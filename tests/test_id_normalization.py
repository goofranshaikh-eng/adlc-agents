from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app
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
