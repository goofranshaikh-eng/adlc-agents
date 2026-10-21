from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_llm_unavailable():
    client = TestClient(app)

    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("LLM API down")

        response = client.post("/query", json={"question": "hello"})

        assert response.status_code == 200
        assert response.json() == {"error": "LLM unavailable"}
