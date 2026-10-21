import os
import re
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError

# Same headers the health probe sends. FastMCP's SSE endpoint expects GET
# with Accept: text/event-stream; tool calls go to the session URL it advertises.
SSE_HEADERS = {"Accept": "text/event-stream"}
_RPC_TIMEOUT_SECONDS = 10.0
_TIMEOUT = timedelta(seconds=_RPC_TIMEOUT_SECONDS)


async def call_tool(tool_name: str, tool_args: dict) -> dict:
    """Call an MCP tool over FastMCP SSE. Never falls back to stdio."""
    mcp_url = os.getenv("MCP_URL", "")
    if not mcp_url:
        return {"error": "MCP_URL not configured"}

    normalized_args = _normalize_id(tool_args or {})

    try:
        async with sse_client(
            mcp_url,
            headers=SSE_HEADERS,
            timeout=_RPC_TIMEOUT_SECONDS,
            sse_read_timeout=_RPC_TIMEOUT_SECONDS,
        ) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=_TIMEOUT
            ) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name,
                    normalized_args,
                    read_timeout_seconds=_TIMEOUT,
                )
                return _unwrap_tool_result(result)
    except Exception as exc:
        return _classify_error(exc)


def _unwrap_tool_result(result) -> dict:
    if getattr(result, "isError", False):
        return {"error": _content_text(getattr(result, "content", None)) or "Tool error"}

    content = getattr(result, "content", None)
    if content:
        return {"result": _content_text(content)}
    return {"result": result}


def _content_text(content) -> str:
    if not content:
        return ""
    block = content[0]
    if isinstance(block, dict):
        return str(block.get("text", ""))
    text = getattr(block, "text", "")
    return "" if text is None else str(text)


def _classify_error(exc: BaseException) -> dict:
    leaves = list(_leaves(exc))
    if any(isinstance(item, (ConnectionError, httpx.ConnectError)) for item in leaves):
        return {"error": "MCP server unreachable"}
    if any(
        isinstance(item, (TimeoutError, httpx.TimeoutException)) for item in leaves
    ):
        return {"error": "MCP server timeout"}
    for item in leaves:
        if isinstance(item, McpError):
            message = str(item).strip() or "Tool error"
            if "timed out" in message.lower():
                return {"error": "MCP server timeout"}
            return {"error": message}
    return {"error": f"MCP client error: {exc}"}


def _leaves(exc: BaseException):
    nested = getattr(exc, "exceptions", None)
    if nested:
        for child in nested:
            yield from _leaves(child)
        return
    yield exc


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
