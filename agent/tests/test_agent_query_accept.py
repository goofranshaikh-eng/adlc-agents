"""AC-1 and AC-2 tests for POST /agent/query."""

import asyncio
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.main import app, run_store

_HEX_RUN_ID = r"^[0-9a-f]{32}$"


def _assert_run_id(value: str) -> None:
    assert isinstance(value, str)
    assert value
    assert len(value) == 32
    parsed = uuid.UUID(value)
    assert parsed.version == 4
    assert value == parsed.hex


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_ac1_accept_valid_query_returns_ok_with_agent_run_id(client):
    response = client.post("/agent/query", json={"query": "What is the weather?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    _assert_run_id(body["agent_run_id"])


def test_ac1_returns_immediately_without_waiting_for_background_job(client):
    t_start = time.perf_counter()
    response = client.post("/agent/query", json={"query": "What is the weather?"})
    elapsed = time.perf_counter() - t_start

    assert response.status_code == 200
    assert elapsed < 0.1

    agent_run_id = response.json()["agent_run_id"]
    run = asyncio.run(run_store.get(agent_run_id))
    assert run is not None
    assert run.status == "pending"


def test_ac2_empty_query_returns_rejected(client):
    response = client.post("/agent/query", json={"query": ""})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "invalid query"
    _assert_run_id(body["agent_run_id"])


def test_ac2_missing_query_field_returns_rejected(client):
    response = client.post("/agent/query", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "invalid query"
    _assert_run_id(body["agent_run_id"])


def test_ac2_whitespace_only_query_returns_rejected(client):
    response = client.post("/agent/query", json={"query": "   \t\n  "})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "invalid query"
    _assert_run_id(body["agent_run_id"])


def test_ac2_rejection_does_not_spawn_background_job(client):
    rejected_payloads = (
        {},
        {"query": ""},
        {"query": "   \t\n  "},
        {"query": 123},
    )

    with (
        patch("agent.main.asyncio.create_task") as mock_create_task,
        patch("agent.main.run_background_job", new=MagicMock()) as mock_job,
    ):
        for payload in rejected_payloads:
            response = client.post("/agent/query", json=payload)
            assert response.status_code == 200
            assert response.json()["status"] == "rejected"

        assert mock_create_task.call_count == 0
        assert mock_job.call_count == 0

        accepted = client.post("/agent/query", json={"query": "What is the weather?"})
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "ok"
        assert mock_create_task.call_count == 1
        assert mock_job.call_count == 1


def test_agent_run_id_unique_across_accept_and_reject(client):
    accepted = client.post("/agent/query", json={"query": "What is the weather?"})
    empty = client.post("/agent/query", json={"query": ""})
    missing = client.post("/agent/query", json={})

    ids = [
        accepted.json()["agent_run_id"],
        empty.json()["agent_run_id"],
        missing.json()["agent_run_id"],
    ]
    for agent_run_id in ids:
        _assert_run_id(agent_run_id)
    assert len(set(ids)) == 3


def test_ac2_rejection_enqueues_failed_event_before_response(client):
    response = client.post("/agent/query", json={"query": ""})
    assert response.status_code == 200
    agent_run_id = response.json()["agent_run_id"]

    run = asyncio.run(run_store.get(agent_run_id))
    assert run is not None
    assert run.id == agent_run_id
    assert run.status == "failed"

    first = run.events.get_nowait()
    second = run.events.get_nowait()
    assert first == {"event": "in_progress", "data": {"run_id": agent_run_id}}
    assert second == {
        "event": "failed",
        "data": {"run_id": agent_run_id, "error": "invalid query"},
    }
    with pytest.raises(asyncio.QueueEmpty):
        run.events.get_nowait()
