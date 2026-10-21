from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_query_direct_answer():
    client = TestClient(app)

    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "type": "direct_answer",
            "answer": "Hello! How can I help you?",
        }

        response = client.post("/query", json={"question": "hello"})

        assert response.status_code == 200
        assert response.json() == {"answer": "Hello! How can I help you?"}

        mock_llm.assert_called_once()
        args = mock_llm.call_args
        assert args[0][0] == "hello"
        assert args[1].get("tool_result") is None
