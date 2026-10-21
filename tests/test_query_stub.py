from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app


def test_query_stays_up():
    client = TestClient(app)

    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "type": "direct_answer",
            "answer": "ok",
        }
        for i in range(5):
            response = client.post("/query", json={"question": f"test {i}"})
            assert response.status_code == 200
