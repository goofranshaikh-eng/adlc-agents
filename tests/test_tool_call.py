"""AC-2: tool calls use the official FastMCP SSE transport.

These tests start the ``mcp`` library's FastMCP SSE server and leave
``sse_client`` unpatched. The agent has to open ``GET /sse`` with
``Accept: text/event-stream``, then post ``initialize``,
``notifications/initialized``, and ``tools/call`` to the session URL.
A JSON-RPC POST straight at the SSE URL does not complete that handshake.
"""

import json
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
    """Record each request as it arrives, including the JSON-RPC method.

    The SSE response stays streaming. Only the request body is copied as the
    server reads it, so the event stream is not buffered.
    """

    def __init__(self, asgi_app, events: list):
        self._app = asgi_app
        self._events = events

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = {
            key.decode().lower(): value.decode()
            for key, value in scope.get("headers", [])
        }
        event = {
            "method": scope["method"],
            "path": scope["path"],
            "accept": headers.get("accept"),
        }
        self._events.append(event)
        body = bytearray()

        async def sniff():
            message = await receive()
            if message["type"] == "http.request":
                body.extend(message.get("body") or b"")
                if not message.get("more_body", False):
                    _attach_rpc(event, bytes(body))
            return message

        await self._app(scope, sniff, send)


def _attach_rpc(event: dict, body: bytes) -> None:
    if event["method"] != "POST" or not body:
        return
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    event["rpc"] = payload.get("method")
    event["params"] = payload.get("params")


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


def _assert_sse_contract(events: list, tool_name: str, arguments: dict) -> None:
    """Fail unless the client completed a real FastMCP SSE session.

    FastMCP ignores a JSON-RPC POST to the SSE URL. The session is a GET on
    ``/sse``, then POSTs to the advertised ``/messages`` endpoint.
    """
    assert events, "client made no HTTP calls to the FastMCP server"
    opened = events[0]
    assert opened["method"] == "GET", "FastMCP SSE requires GET before any POST"
    assert opened["path"].rstrip("/") == "/sse"
    assert "text/event-stream" in (opened["accept"] or "")
    assert all(
        not (event["method"] == "POST" and event["path"].rstrip("/") == "/sse")
        for event in events
    ), "client posted JSON-RPC to the SSE URL; FastMCP rejects that"

    posts = [event for event in events if event["method"] == "POST"]
    assert posts, "client never posted a JSON-RPC message"
    assert all(
        event["path"].rstrip("/") == "/messages" for event in posts
    ), "tool calls must POST to the SSE session URL, not the SSE URL"
    methods = [event.get("rpc") for event in posts]
    assert "initialize" in methods
    assert "notifications/initialized" in methods
    call_indexes = [index for index, method in enumerate(methods) if method == "tools/call"]
    assert call_indexes, f"client never called a tool; methods were {methods}"
    assert methods.index("initialize") < methods.index("notifications/initialized") < call_indexes[-1]

    call = posts[call_indexes[-1]]
    assert call["params"]["name"] == tool_name
    assert call["params"]["arguments"] == arguments


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
        _assert_sse_contract(peer["events"], "list_sales_order", {})
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
        _assert_sse_contract(peer["events"], tool_name, expected_args)
        assert peer["calls"] == [(tool_name, expected_args)]
        tool_result = mock_llm.await_args_list[1].kwargs["tool_result"]
        assert tool_result == {"result": "ok"}
