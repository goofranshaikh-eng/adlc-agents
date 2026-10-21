"""AC-2: tool calls use the official FastMCP SSE transport.

A test that mocks ``httpx.post`` and returns 200 still passes if the agent
POSTs JSON-RPC at the SSE URL. FastMCP rejects that. These tests start the
``mcp`` library's FastMCP SSE server and leave ``sse_client`` and ``httpx``
unpatched, so the handshake has to succeed for real.
"""

import threading
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from fastapi.testclient import TestClient
from mcp.client.sse import sse_client as official_sse_client
from mcp.server.fastmcp import FastMCP

from agent import mcp_client
from agent.main import app


def test_transport_uses_official_sse_client():
    assert mcp_client.sse_client is official_sse_client


class _RecordingASGI:
    """Record the HTTP method and path without buffering the SSE stream."""

    def __init__(self, asgi_app, events: list):
        self._app = asgi_app
        self._events = events

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
            self._events.append(
                {
                    "method": scope["method"],
                    "path": scope["path"],
                    "accept": headers.get("accept"),
                }
            )
        await self._app(scope, receive, send)


@contextmanager
def live_fastmcp():
    """Serve the four AC-2 tools on the real FastMCP SSE transport."""
    calls = []
    events = []
    server_app = FastMCP("ac2-transport", host="127.0.0.1", port=0, log_level="ERROR")

    @server_app.tool()
    def list_sales_order() -> str:
        calls.append(("list_sales_order", {}))
        return "ok"

    @server_app.tool()
    def list_employees() -> str:
        calls.append(("list_employees", {}))
        return "ok"

    @server_app.tool()
    def get_employee_by_id(id: str) -> str:
        calls.append(("get_employee_by_id", {"id": id}))
        return "ok"

    @server_app.tool()
    def get_sales_order_by_id(id: str) -> str:
        calls.append(("get_sales_order_by_id", {"id": id}))
        return "ok"

    config = uvicorn.Config(
        _RecordingASGI(server_app.sse_app(), events),
        host="127.0.0.1",
        port=0,
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("FastMCP SSE server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield {"url": f"http://127.0.0.1:{port}/sse", "calls": calls, "events": events}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _assert_sse_contract(events: list) -> None:
    assert events, "client made no HTTP calls to the FastMCP server"
    opened = events[0]
    assert opened["method"] == "GET", "FastMCP SSE requires GET before any POST"
    assert opened["path"].rstrip("/") == "/sse"
    assert "text/event-stream" in (opened["accept"] or "")
    assert all(
        not (event["method"] == "POST" and event["path"].rstrip("/") == "/sse") for event in events
    ), "client posted JSON-RPC to the SSE URL; FastMCP rejects that"
    assert any(event["method"] == "POST" and event["path"].rstrip("/") == "/messages" for event in events)


def test_tool_call_success(monkeypatch):
    with live_fastmcp() as peer:
        monkeypatch.setenv("MCP_URL", peer["url"])
        client = TestClient(app)

        with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                {
                    "type": "tool_request",
                    "tool_name": "list_sales_order",
                    "tool_args": {},
                },
                {
                    "type": "direct_answer",
                    "answer": "Here are the sales orders: SO-001 ($100)",
                },
            ]

            response = client.post("/query", json={"question": "list sales orders"})

        assert response.status_code == 200
        assert "sales orders" in response.json()["answer"].lower()
        _assert_sse_contract(peer["events"])
        assert peer["calls"] == [("list_sales_order", {})]
        tool_result = mock_llm.await_args_list[1].kwargs["tool_result"]
        assert tool_result == {"result": "ok"}


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected_args"),
    [
        ("list_sales_order", {}, {}),
        ("list_employees", {}, {}),
        ("get_employee_by_id", {"id": "emp-001"}, {"id": "EMP-001"}),
        ("get_sales_order_by_id", {"id": "so-123"}, {"id": "SO-123"}),
    ],
)
def test_four_tools_use_fastmcp_sse(monkeypatch, tool_name, tool_args, expected_args):
    with live_fastmcp() as peer:
        monkeypatch.setenv("MCP_URL", peer["url"])
        client = TestClient(app)

        with patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                {
                    "type": "tool_request",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
                {"type": "direct_answer", "answer": f"called {tool_name}"},
            ]
            response = client.post("/query", json={"question": tool_name})

        assert response.status_code == 200
        assert tool_name in response.json()["answer"]
        _assert_sse_contract(peer["events"])
        assert peer["calls"] == [(tool_name, expected_args)]
        tool_result = mock_llm.await_args_list[1].kwargs["tool_result"]
        assert tool_result == {"result": "ok"}
