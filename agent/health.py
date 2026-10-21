import os
from datetime import datetime, timedelta

import httpx

from agent.mcp_client import SSE_HEADERS

_probe_cache = {"reachable": False, "timestamp": None}
_CACHE_TTL_SECONDS = 10


async def check_health() -> dict:
    mcp_url = os.getenv("MCP_URL", "")

    if not mcp_url:
        return {"status": "up", "mcp_reachable": False}

    now = datetime.now()
    timestamp = _probe_cache["timestamp"]
    if timestamp and (now - timestamp) < timedelta(seconds=_CACHE_TTL_SECONDS):
        return {"status": "up", "mcp_reachable": _probe_cache["reachable"]}

    reachable = await _probe_mcp(mcp_url)
    _probe_cache["reachable"] = reachable
    _probe_cache["timestamp"] = now

    return {"status": "up", "mcp_reachable": reachable}


async def _probe_mcp(url: str) -> bool:
    """Reachability check aligned with the FastMCP SSE client.

    FastMCP SSE endpoints expect GET with ``Accept: text/event-stream``.
    HEAD often hangs or is rejected, so a HEAD probe reports a live server
    as unreachable. Status is read from the response headers and the stream
    is closed without calling a tool.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            async with client.stream("GET", url, headers=SSE_HEADERS) as response:
                return response.status_code < 500
    except Exception:
        return False
