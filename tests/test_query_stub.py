from fastapi.testclient import TestClient

from agent.main import app


def test_query_stub_response():
    client = TestClient(app)
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "Placeholder" in response.json()["answer"]


def test_query_stays_up():
    client = TestClient(app)
    for i in range(5):
        response = client.post("/query", json={"question": f"test {i}"})
        assert response.status_code == 200
