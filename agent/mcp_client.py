import json
import os
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
    """Return ``{"error": ...}`` for protocol errors and tool payloads.

    FastMCP tools report failure as a normal result whose body is
    ``{"error": "..."}`` (``isError`` stays false). That shape must not be
    wrapped as a successful result for the LLM to paraphrase.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)

    content = getattr(result, "content", None)
    text = _content_text(content) if content else ""
    payload_error = _payload_error(structured) or _payload_error(_parse_json_object(text))

    if getattr(result, "isError", False):
        return {"error": payload_error or text or "Tool error"}

    if payload_error:
        return {"error": payload_error}

    if content:
        return {"result": text}
    return {"result": result}


def _parse_json_object(text: str):
    if not text or not text.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _payload_error(payload) -> str | None:
    """Extract a clear error from a tool body of ``{"error": "..."}``."""
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    if any(key in payload for key in ("salesOrders", "employees", "salesOrder", "employee")):
        return None

    error = payload["error"]
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return str(error)


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
    """Uppercase an ``id`` argument the way the stdio agent did.

    Audit of the original stdio orchestrator: the only normalization is
    ``value.upper()`` when the argument key is ``id``. There is no format
    regex, so hyphenated and unhyphenated ids are both supported. The hyphen,
    underscore, and extra segments are kept. Patterns that rule covers:

    - ``emp-001``, ``Emp-001``, ``eMp-001``, ``EMP-001`` → ``EMP-001``
    - ``emp001``, ``Emp001``, ``EMP001`` → ``EMP001``
    - ``so-123``, ``SO-123`` → ``SO-123``
    - ``so123`` → ``SO123``
    - ``ord-001``, ``Ord-001`` → ``ORD-001``
    - ``foo-bar-123`` → ``FOO-BAR-123``
    - ``emp_001`` → ``EMP_001``

    Other argument keys are left unchanged. Non-string ``id`` values are left
    unchanged (the original only received strings; ``.upper()`` on a non-string
    would raise).
    """
    if "id" not in args:
        return args

    id_value = args["id"]
    if not isinstance(id_value, str):
        return args

    return {**args, "id": id_value.upper()}
