"""AC-6 regression tests for POST /query and GET /health.

Baselines are the pre-Stories-1–2 contract for this repository:

- POST /query blocks on ``process_question`` and returns its dict unchanged.
- GET /health is exactly ``{"status": "ok"}`` and does not touch the run store.
- This worktree does not vendor PER-20's orchestrator. Tool names and id
  normalization are unchanged when those modules are absent, and match the
  PER-20 set when they are present.
- MCP_URL is unset in repo config (PER-20 reads ``os.getenv("MCP_URL", "")``).
"""

import asyncio
import importlib
import os
import re
import statistics
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.main import app, run_store

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_PKG = REPO_ROOT / "agent"

# Mean of a fast blocking POST /query before Stories 1–2 added a request-path
# lock or other synchronous work. Tolerance is 10% on the mean and 20% per call.
BASELINE_QUERY_LATENCY_MS = 100.0

# PER-20 agent/llm_client.py _TOOLS. Not vendored in this worktree.
PER20_TOOL_NAMES = frozenset(
    {
        "list_sales_order",
        "list_employees",
        "get_sales_order_by_id",
        "get_employee_by_id",
    }
)

# PER-20 agent/mcp_client.py _normalize_id: uppercase a string id only.
BASELINE_ID_CASES = (
    ({"id": "emp-001"}, {"id": "EMP-001"}),
    ({"id": "ord-001"}, {"id": "ORD-001"}),
    ({"id": "emp001"}, {"id": "EMP001"}),
    ({"id": "foo-bar-123"}, {"id": "FOO-BAR-123"}),
    ({"id": "emp_001"}, {"id": "EMP_001"}),
    ({"id": "emp-001", "note": "keep"}, {"id": "EMP-001", "note": "keep"}),
    ({"limit": 10}, {"limit": 10}),
    ({"id": 1}, {"id": 1}),
)

# No .env and no hardcoded URL. PER-20 default is an empty string.
BASELINE_MCP_URL = ""

_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z0-9_]+)"')
_NORMALIZE_DEF_RE = re.compile(r"def\s+(?:_normalize_id|normalize_id|slugify)\b")
_MCP_URL_ASSIGN_RE = re.compile(
    r"""MCP_URL["']?\s*[:=]\s*["']([^"']+)["']"""
)
_THREAD_OR_PROCESS_RE = re.compile(
    r"\b(?:threading\.Thread|ThreadPoolExecutor|multiprocessing\.|os\.fork)\b"
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def orchestrator():
    """Install a fake PER-20 orchestrator for the duration of one test."""
    previous = sys.modules.get("agent.orchestrator")
    module = types.ModuleType("agent.orchestrator")
    sys.modules["agent.orchestrator"] = module
    yield module
    if previous is None:
        sys.modules.pop("agent.orchestrator", None)
    else:
        sys.modules["agent.orchestrator"] = previous


def _agent_sources() -> list[Path]:
    return [
        path
        for path in AGENT_PKG.rglob("*.py")
        if "tests" not in path.parts and path.name != "conftest.py"
    ]


def _child_pids() -> set[int]:
    """Child process ids of this process. Uses psutil when it is installed."""
    try:
        import psutil
    except ImportError:
        found: set[int] = set()
        task_dir = Path(f"/proc/{os.getpid()}/task")
        if not task_dir.is_dir():
            return found
        for tid in task_dir.iterdir():
            children = tid / "children"
            if not children.is_file():
                continue
            for token in children.read_text().split():
                found.add(int(token))
        return found

    return {child.pid for child in psutil.Process().children(recursive=False)}


def _extract_tool_names() -> set[str]:
    """Tool names from orchestrator.py or llm_client.py, if those modules exist."""
    names: set[str] = set()
    for relative in ("orchestrator.py", "llm_client.py"):
        path = AGENT_PKG / relative
        if path.is_file():
            names.update(_TOOL_NAME_RE.findall(path.read_text()))
    return names


def _registry_present() -> bool:
    return (AGENT_PKG / "orchestrator.py").is_file() or (
        AGENT_PKG / "llm_client.py"
    ).is_file()


def _configured_mcp_url() -> str:
    """MCP_URL from the environment or a repo config file. Default is empty."""
    for candidate in (
        REPO_ROOT / ".env",
        REPO_ROOT / "agent" / ".env",
        REPO_ROOT / "config.py",
        REPO_ROOT / "agent" / "config.py",
    ):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("MCP_URL="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    return os.getenv("MCP_URL", "")


def test_ac6_post_query_valid_question_returns_blocking_answer(client, orchestrator):
    delay_s = 0.2
    calls: list[tuple[float, float, str]] = []

    async def process_question(question: str) -> dict:
        started = time.perf_counter()
        await asyncio.sleep(delay_s)
        finished = time.perf_counter()
        calls.append((started, finished, question))
        return {"answer": "sunny"}

    orchestrator.process_question = process_question

    started = time.perf_counter()
    response = client.post("/query", json={"question": "What is the weather?"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.json() == {"answer": "sunny"}
    assert len(calls) == 1
    _start, _finish, question = calls[0]
    assert question == "What is the weather?"
    mock_latency = calls[0][1] - calls[0][0]
    assert elapsed >= mock_latency - 0.02


def test_ac6_post_query_error_returns_blocking_error(client, orchestrator):
    delay_s = 0.2
    calls: list[float] = []

    async def process_question(question: str) -> dict:
        started = time.perf_counter()
        await asyncio.sleep(delay_s)
        calls.append(time.perf_counter() - started)
        return {"error": "LLM unavailable"}

    orchestrator.process_question = process_question

    started = time.perf_counter()
    response = client.post("/query", json={"question": "hello"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.json() == {"error": "LLM unavailable"}
    assert len(calls) == 1
    assert elapsed >= calls[0] - 0.02


def test_ac6_post_query_latency_within_baseline(client, orchestrator):
    async def process_question(question: str) -> dict:
        return {"answer": question}

    orchestrator.process_question = process_question

    samples_ms: list[float] = []
    for index in range(10):
        started = time.perf_counter()
        response = client.post("/query", json={"question": f"question {index}"})
        samples_ms.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200
        assert response.json() == {"answer": f"question {index}"}

    mean_post = statistics.mean(samples_ms)
    assert mean_post <= BASELINE_QUERY_LATENCY_MS * 1.1
    for sample in samples_ms:
        assert sample <= BASELINE_QUERY_LATENCY_MS * 1.2


def test_ac6_get_health_returns_ok_without_side_effects(client):
    accepted = client.post("/agent/query", json={"query": "What is the weather?"})
    assert accepted.status_code == 200
    agent_run_id = accepted.json()["agent_run_id"]
    run_before = asyncio.run(run_store.get(agent_run_id))
    assert run_before is not None
    status_before = run_before.status
    queued_before = run_before.events.qsize()
    stored_before = set(run_store._runs)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    run_after = asyncio.run(run_store.get(agent_run_id))
    assert run_after is not None
    assert run_after.status == status_before
    assert run_after.events.qsize() == queued_before
    assert set(run_store._runs) == stored_before


def test_ac6_post_query_and_agent_query_coexist(client, orchestrator):
    delay_s = 0.2

    async def process_question(question: str) -> dict:
        await asyncio.sleep(delay_s)
        return {"answer": "blocked-answer"}

    orchestrator.process_question = process_question

    agent_started = time.perf_counter()
    accepted = client.post("/agent/query", json={"query": "What is the weather?"})
    agent_elapsed = time.perf_counter() - agent_started

    assert accepted.status_code == 200
    assert agent_elapsed < 0.1
    body = accepted.json()
    assert body["status"] == "ok"
    agent_run_id = body["agent_run_id"]

    query_started = time.perf_counter()
    queried = client.post("/query", json={"question": "What is the weather?"})
    query_elapsed = time.perf_counter() - query_started

    assert queried.status_code == 200
    assert queried.json() == {"answer": "blocked-answer"}
    assert query_elapsed >= delay_s - 0.02

    run = asyncio.run(run_store.get(agent_run_id))
    assert run is not None
    assert run.status == "pending"
    first = run.events.get_nowait()
    assert first == {"event": "in_progress", "data": {"run_id": agent_run_id}}

    route_paths = {getattr(route, "path", None) for route in app.routes}
    assert "/query" in route_paths
    assert "/agent/query" in route_paths
    assert not (AGENT_PKG / "routes").exists()
    assert not (AGENT_PKG / "routers").exists()
    assert not (AGENT_PKG / "src").exists()


def test_ac6_mcp_called_only_from_main_process(client, orchestrator):
    main_pid = os.getpid()
    seen: list[int] = []
    threads_during: list[set[int]] = []

    async def process_question(question: str) -> dict:
        seen.append(os.getpid())
        threads_during.append({thread.ident for thread in threading.enumerate()})
        return {"answer": "ok"}

    orchestrator.process_question = process_question

    children_before = _child_pids()
    threads_before = {thread.ident for thread in threading.enumerate()}

    accepted = client.post("/agent/query", json={"query": "What is the weather?"})
    queried = client.post("/query", json={"question": "What is the weather?"})

    assert accepted.status_code == 200
    assert queried.status_code == 200
    assert seen == [main_pid]
    assert _child_pids() == children_before
    assert threads_during == [threads_before]

    for path in _agent_sources():
        source = path.read_text()
        assert _THREAD_OR_PROCESS_RE.search(source) is None
    main_source = (AGENT_PKG / "main.py").read_text()
    assert "asyncio.create_task(" in main_source


def test_ac6_tool_names_unchanged():
    current = _extract_tool_names()
    if _registry_present():
        assert current == PER20_TOOL_NAMES
    else:
        # Stories 1–2 did not add orchestrator.py or llm_client.py.
        assert current == frozenset()
        assert not (AGENT_PKG / "orchestrator.py").exists()
        assert not (AGENT_PKG / "llm_client.py").exists()


def test_ac6_id_normalization_unchanged():
    introduced = [
        path.name
        for path in _agent_sources()
        if path.name != "mcp_client.py" and _NORMALIZE_DEF_RE.search(path.read_text())
    ]
    assert introduced == []

    try:
        mcp_client = importlib.import_module("agent.mcp_client")
    except ImportError:
        mcp_client = None

    if mcp_client is None or not hasattr(mcp_client, "_normalize_id"):
        return

    normalize = mcp_client._normalize_id
    for raw, expected in BASELINE_ID_CASES:
        assert normalize(dict(raw)) == expected


def test_ac6_mcp_url_unchanged(monkeypatch):
    monkeypatch.delenv("MCP_URL", raising=False)
    assert _configured_mcp_url() == BASELINE_MCP_URL

    for path in _agent_sources():
        assert _MCP_URL_ASSIGN_RE.search(path.read_text()) is None
