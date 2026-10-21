import os
from datetime import datetime, timedelta

import httpx

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
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.head(url)
            return response.status_code < 500
    except Exception:
        return False
