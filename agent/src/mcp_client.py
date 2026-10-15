import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, server_url: Optional[str] = None):
        if server_url is None:
            server_url = os.environ.get("MCP_SERVER_URL")

        if not server_url:
            raise ValueError("MCP server URL not configured")

        self._server_url = server_url
        self._request_id = 0

    def _invoke_tool(self, tool_name: str, params: dict) -> Optional[dict]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": self._request_id
        }

        try:
            response = requests.post(
                self._server_url,
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"Tool {tool_name} returned HTTP {response.status_code}")
                return None

            data = response.json()

            if "error" in data:
                logger.warning(f"Tool {tool_name} returned error: {data['error']}")
                return None

            if "result" not in data:
                logger.warning(f"Tool {tool_name} returned malformed response")
                return None

            return data["result"]

        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error calling tool {tool_name}: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.warning(f"Parse error for tool {tool_name}: {e}")
            return None

    def invoke_list_sales_order(self) -> Optional[dict]:
        return self._invoke_tool("list_sales_order", {})

    def invoke_list_employees(self) -> Optional[dict]:
        return self._invoke_tool("list_employees", {})

    def invoke_get_sales_order_by_id(self, order_id: str) -> Optional[dict]:
        return self._invoke_tool("get_sales_order_by_id", {"id": order_id})

    def invoke_get_employee_by_id(self, employee_id: str) -> Optional[dict]:
        return self._invoke_tool("get_employee_by_id", {"id": employee_id})
