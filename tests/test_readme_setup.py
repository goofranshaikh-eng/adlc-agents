"""AC-5: README documents the FastAPI primary path and omits Flask/stdio."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.main import app

README_PATH = Path(__file__).resolve().parents[1] / "README.md"

_REMOVED_PATTERNS = ("flask", "stdio", "subprocess", "local mcp", "stdio_mcp")

_REQUIRED_SECTIONS = (
    "# TriNet Agent",
    "## Prerequisites",
    "## Installation",
    "## Configuration",
    "## Running the Agent",
    "## Usage",
    "## Health Check",
    "## Development",
)


def _readme() -> str:
    return README_PATH.read_text()


def test_readme_no_flask_or_stdio():
    """README does not document Flask or stdio MCP."""
    content = _readme().lower()

    for pattern in _REMOVED_PATTERNS:
        assert pattern not in content, f"README still references {pattern}"


def test_readme_documents_fastapi_primary_path():
    """Setup docs tell a developer to run FastAPI, set MCP_URL and LLM env, and POST a question."""
    content = _readme()
    lowered = content.lower()

    positions = []
    for heading in _REQUIRED_SECTIONS:
        index = content.find(heading)
        assert index != -1, f"README missing section {heading}"
        positions.append(index)
    assert positions == sorted(positions)

    assert "fastapi" in lowered
    assert "mcp_url" in lowered
    assert "http://127.0.0.1:8000/sse" in content
    assert "anthropic_api_key" in lowered
    assert "llm_gateway_url" in lowered
    assert "uvicorn agent.main:app --host 0.0.0.0 --port 8001" in content
    assert "post /query" in lowered or 'post http://localhost:8001/query' in lowered
    assert "/health" in content
    assert "pytest tests/ -v" in content


def test_readme_setup_end_to_end(monkeypatch):
    """Follow the documented env and POST /query contract.

    The manual procedure starts uvicorn and curls the server. This check uses
    the same HTTP contract in-process with a mock LLM, which is what CI can
    run without a live Anthropic key or MCP process.
    """
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8000/sse")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    client = TestClient(app)
    with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "type": "direct_answer",
            "answer": "Hello! How can I help you?",
        }
        response = client.post("/query", json={"question": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Hello! How can I help you?"
    assert len(data["answer"]) > 0
