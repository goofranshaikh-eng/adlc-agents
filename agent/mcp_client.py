import os
import re

import httpx


async def call_tool(tool_name: str, tool_args: dict) -> dict:
    """Call an MCP tool over FastMCP HTTP (SSE). Never falls back to stdio."""
    mcp_url = os.getenv("MCP_URL", "")
    if not mcp_url:
        return {"error": "MCP_URL not configured"}

    normalized_args = _normalize_id(tool_args)

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": normalized_args,
        },
        "id": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(mcp_url, json=payload)
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                error = data["error"]
                if isinstance(error, dict):
                    return {"error": error.get("message", "Tool error")}
                return {"error": str(error)}

            return {"result": data.get("result", {})}

    except httpx.ConnectError:
        return {"error": "MCP server unreachable"}
    except httpx.TimeoutException:
        return {"error": "MCP server timeout"}
    except httpx.HTTPStatusError as e:
        return {"error": f"MCP server error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"MCP client error: {str(e)}"}


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
