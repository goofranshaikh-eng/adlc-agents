import asyncio
import contextlib
import json
import os
import re
from urllib.parse import urljoin, urlparse

import httpx

# FastMCP's SSE transport is a GET that stays open. Tool calls are JSON-RPC
# POSTs to the session endpoint advertised on that stream, not POSTs to the
# SSE URL itself.
SSE_HEADERS = {"Accept": "text/event-stream"}
_RPC_TIMEOUT_SECONDS = 10.0


async def call_tool(tool_name: str, tool_args: dict) -> dict:
    """Call an MCP tool over FastMCP SSE. Never falls back to stdio."""
    mcp_url = os.getenv("MCP_URL", "")
    if not mcp_url:
        return {"error": "MCP_URL not configured"}

    normalized_args = _normalize_id(tool_args)

    try:
        async with httpx.AsyncClient(timeout=_RPC_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", mcp_url, headers=SSE_HEADERS) as response:
                response.raise_for_status()
                return await _call_over_sse(
                    client, mcp_url, response, tool_name, normalized_args
                )
    except httpx.ConnectError:
        return {"error": "MCP server unreachable"}
    except (httpx.TimeoutException, asyncio.TimeoutError):
        return {"error": "MCP server timeout"}
    except httpx.HTTPStatusError as e:
        return {"error": f"MCP server error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"MCP client error: {str(e)}"}


async def _call_over_sse(client, mcp_url: str, response, tool_name: str, arguments: dict) -> dict:
    loop = asyncio.get_running_loop()
    endpoint_future = loop.create_future()
    pending: dict[int, asyncio.Future] = {}
    next_id = 1

    reader_task = asyncio.create_task(_read_sse(response, endpoint_future, pending))

    def _fail_waiters(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception() or RuntimeError("SSE stream closed")
        if not endpoint_future.done():
            endpoint_future.set_exception(exc)
            return
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)
                return

    reader_task.add_done_callback(_fail_waiters)

    try:
        endpoint = await asyncio.wait_for(endpoint_future, _RPC_TIMEOUT_SECONDS)
        message_url = _message_url(mcp_url, endpoint)

        async def rpc(method: str, params: dict | None = None, *, notify: bool = False):
            nonlocal next_id
            payload = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                payload["params"] = params
            waiter = None
            if not notify:
                payload["id"] = next_id
                waiter = loop.create_future()
                pending[next_id] = waiter
                next_id += 1
            post = await client.post(message_url, json=payload)
            post.raise_for_status()
            if notify:
                return None
            return await asyncio.wait_for(waiter, _RPC_TIMEOUT_SECONDS)

        initialized = await rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "trinet-agent", "version": "0.1.0"},
            },
        )
        if "error" in initialized:
            return _rpc_error(initialized)
        await rpc("notifications/initialized", {}, notify=True)
        result = await rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        return _unwrap_tool_result(result)
    finally:
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader_task


async def _read_sse(response, endpoint_future, pending: dict) -> None:
    event_name = "message"
    data_lines: list[str] = []

    def _finish_event() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return
        data = "\n".join(data_lines)
        if event_name == "endpoint" and not endpoint_future.done():
            endpoint_future.set_result(data.strip())
        elif event_name == "message":
            _dispatch_message(data, pending)
        event_name = "message"
        data_lines = []

    async for line in response.aiter_lines():
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            _finish_event()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value.strip()
        elif field == "data":
            data_lines.append(value)
    _finish_event()


def _dispatch_message(data: str, pending: dict) -> None:
    try:
        message = json.loads(data)
    except json.JSONDecodeError:
        return
    msg_id = message.get("id")
    fut = pending.get(msg_id)
    if fut is not None and not fut.done():
        fut.set_result(message)


def _message_url(sse_url: str, endpoint: str) -> str:
    target = urljoin(sse_url, endpoint.strip())
    base = urlparse(sse_url)
    parsed = urlparse(target)
    if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
        raise ValueError("SSE endpoint origin mismatch")
    return target


def _rpc_error(message: dict) -> dict:
    error = message.get("error")
    if isinstance(error, dict):
        return {"error": error.get("message", "Tool error")}
    return {"error": str(error)}


def _unwrap_tool_result(message: dict) -> dict:
    if "error" in message:
        return _rpc_error(message)

    result = message.get("result", {})
    if isinstance(result, dict) and result.get("isError"):
        texts = [
            block.get("text", "")
            for block in result.get("content") or []
            if isinstance(block, dict) and block.get("text")
        ]
        return {"error": " ".join(texts) or "Tool error"}
    return {"result": result}


def _normalize_id(args: dict) -> dict:
    """Normalize 'id' argument: emp-001 → EMP-001, so-123 → SO-123."""
    if "id" not in args:
        return args

    id_value = args["id"]
    if not isinstance(id_value, str):
        return args

    match = re.match(r"^([a-z]+)-(\d+)$", id_value, re.IGNORECASE)
    if match:
        prefix, digits = match.groups()
        return {**args, "id": f"{prefix.upper()}-{digits}"}

    return args
