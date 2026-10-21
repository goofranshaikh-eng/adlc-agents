"""In-process FastMCP SSE peer for transport tests.

A JSON-RPC POST to the SSE URL is rejected. Clients must GET `/sse`, read the
endpoint event, initialize the session, and POST messages to that endpoint.
"""

import json
import queue
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class _FastMCPServer(ThreadingHTTPServer):
    def __init__(self, tool_result):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.tool_result = tool_result
        self.posts = []
        self.gets = []
        self.outbound = queue.Queue()
        self.initialized = False
        self.daemon_threads = True

    @property
    def url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/sse"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        return

    def do_HEAD(self):
        self.send_response(501)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        self.server.gets.append({"path": path, "accept": self.headers.get("Accept")})
        if path.rstrip("/") != "/sse":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        host, port = self.server.server_address
        endpoint = f"http://{host}:{port}/messages/?session_id=test"
        try:
            self._write_event("endpoint", endpoint)
            while True:
                message = self.server.outbound.get(timeout=10)
                if message is None:
                    return
                self._write_event("message", json.dumps(message))
        except Exception:
            return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        path = urlparse(self.path).path
        try:
            payload = json.loads(raw.decode() or "null")
        except json.JSONDecodeError:
            payload = {"_raw": raw.decode(errors="replace")}
        self.server.posts.append({"path": path, "body": payload})

        if path.rstrip("/") != "/messages":
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        method = payload.get("method") if isinstance(payload, dict) else None
        if method == "notifications/initialized":
            self.server.initialized = True
            self._accepted()
            return
        if method == "initialize":
            self.server.outbound.put(
                {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-fastmcp", "version": "0.0.1"},
                    },
                }
            )
            self._accepted()
            return
        if method == "tools/list":
            self.server.outbound.put(_tools_list(payload))
            self._accepted()
            return
        if method == "tools/call":
            if not self.server.initialized:
                response = {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "error": {"code": -32002, "message": "Session not initialized"},
                }
            else:
                response = self.server.tool_result(payload)
            self.server.outbound.put(response)
            self._accepted()
            return

        self.send_response(400)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _accepted(self):
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _write_event(self, name: str, data: str):
        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
        self.wfile.flush()


@contextmanager
def fake_fastmcp(tool_result):
    server = _FastMCPServer(tool_result)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.outbound.put(None)
        server.shutdown()
        server.server_close()


_TOOL_NAMES = (
    "list_sales_order",
    "list_employees",
    "get_sales_order_by_id",
    "get_employee_by_id",
)


def _tools_list(payload: dict) -> dict:
    """Answer tools/list so the official SSE client can finish a call."""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {
            "tools": [
                {
                    "name": name,
                    "inputSchema": {"type": "object", "properties": {}},
                }
                for name in _TOOL_NAMES
            ]
        },
    }


def tool_success(payload: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"content": [{"type": "text", "text": "ok"}], "isError": False},
    }


def assert_fastmcp_tool_call(server) -> dict:
    """Return the tools/call body if the client used the SSE session."""
    assert server.gets, "client never opened the SSE endpoint"
    assert "text/event-stream" in (server.gets[0]["accept"] or "")
    assert server.posts, "client never posted a JSON-RPC message"
    assert all(p["path"].rstrip("/") != "/sse" for p in server.posts)
    methods = [p["body"].get("method") for p in server.posts]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/call"]
    tool_calls = [p["body"] for p in server.posts if p["body"].get("method") == "tools/call"]
    assert len(tool_calls) == 1
    return tool_calls[0]
