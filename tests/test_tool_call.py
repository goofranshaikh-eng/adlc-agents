"""AC-2: tool calls use the mcp library SSE client, not a JSON-RPC POST.

The peer rejects ``POST`` of a tools/call body to the SSE URL. A test that
mocks ``httpx.post`` and returns 200 would still pass against that shortcut;
these tests run the real ``mcp.client.sse.sse_client`` instead.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.client.sse import sse_client as mcp_sse_client

from agent.main import app
from tests.fastmcp_fake import assert_fastmcp_tool_call, fake_fastmcp, tool_success


def _track_sse(calls: list):
    """Record sse_client invocations, then perform the real FastMCP handshake."""

    def factory(*args, **kwargs):
        @asynccontextmanager
        async def _cm():
            calls.append((args, kwargs))
            async with mcp_sse_client(*args, **kwargs) as streams:
                yield streams

        return _cm()

    return patch("agent.mcp_client.sse_client", factory)


def _assert_official_sse_client(calls: list, server) -> None:
    assert calls, "agent did not call mcp.client.sse.sse_client"
    args, kwargs = calls[0]
    assert args[0] == server.url
    assert kwargs.get("headers", {}).get("Accept") == "text/event-stream"


def test_jsonrpc_post_to_sse_url_is_rejected():
    """The old httpx.post(MCP_URL, json=rpc) shape is not FastMCP SSE."""
    with fake_fastmcp(tool_success) as server:
        response = httpx.post(
            server.url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "list_sales_order", "arguments": {}},
                "id": 1,
            },
        )

    assert response.status_code == 405
    assert server.posts[0]["path"].rstrip("/") == "/sse"
    assert server.gets == []


def test_tool_call_success(monkeypatch):
    sse_calls = []
    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        with (
            patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm,
            _track_sse(sse_calls),
        ):
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
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == "list_sales_order"
        assert call["params"]["arguments"] == {}
        _assert_official_sse_client(sse_calls, server)
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
    sse_calls = []
    with fake_fastmcp(tool_success) as server:
        monkeypatch.setenv("MCP_URL", server.url)
        client = TestClient(app)

        with (
            patch("agent.llm_client.ask_llm", new_callable=AsyncMock) as mock_llm,
            _track_sse(sse_calls),
        ):
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
        call = assert_fastmcp_tool_call(server)
        assert call["params"]["name"] == tool_name
        assert call["params"]["arguments"] == expected_args
        _assert_official_sse_client(sse_calls, server)
        tool_result = mock_llm.await_args_list[1].kwargs["tool_result"]
        assert tool_result == {"result": "ok"}
