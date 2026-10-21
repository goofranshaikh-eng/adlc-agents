# TriNet Agent

AI agent that answers questions about TriNet employee and sales-order data. The agent uses an LLM (Anthropic Claude or compatible gateway) to decide whether to answer directly or call MCP tools over HTTP. Successful tool data is synthesized into the final answer. A tool result of `{"error": "..."}`, or an unreachable MCP server, is returned as that error.

**Architecture:**
- **FastAPI** web server serving `/query` and `/health` endpoints
- **LLM client** for direct answers and synthesis of successful tool data
- **FastMCP HTTP client** to call MCP tools at a shared MCP server

## Prerequisites

- **Python 3.11+** (or version specified in `pyproject.toml` / `requirements.txt`)
- **MCP Server** running at a known URL (e.g. `http://127.0.0.1:8000/sse`)
  - See [PER-18](https://linear.app/personalgs/issue/PER-18) for MCP server setup
- **LLM credentials** (Anthropic API key or gateway URL)

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd adlc-agents
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Set the following environment variables:

| Variable | Required | Description | Example |
|---|---|---|---|
| `MCP_URL` | Yes | FastMCP SSE endpoint URL | `http://127.0.0.1:8000/sse` |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key for Claude LLM | `sk-ant-...` |
| `LLM_GATEWAY_URL` | Yes* | Alternative: LLM gateway URL if not using Anthropic directly | `http://llm-gateway:8080` |

_*Either `ANTHROPIC_API_KEY` or `LLM_GATEWAY_URL` is required, depending on your LLM setup._

**Example `.env` file:**
```bash
MCP_URL=http://127.0.0.1:8000/sse
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

**Example IDs** (any string `id` is uppercased before the tool call; hyphens and other separators are kept). Unhyphenated ids are included, not dropped:

| Example input | Sent to the tool |
|---|---|
| `emp-001`, `Emp-001`, `eMp-001`, `EMP-001` | `EMP-001` |
| `emp001`, `Emp001`, `EMP001` | `EMP001` |
| `so-123`, `SO-123` | `SO-123` |
| `so123` | `SO123` |
| `ord-001`, `Ord-001` | `ORD-001` |
| `foo-bar-123`, `Foo-Bar-123` | `FOO-BAR-123` |
| `emp_001` | `EMP_001` |

## Running the Agent

Start the FastAPI server with `uvicorn`:

```bash
uvicorn agent.main:app --host 0.0.0.0 --port 8001
```

**Options:**
- `--reload` — auto-reload on code changes (development only)
- `--workers N` — run N worker processes (production)

The agent will be available at `http://localhost:8001`.

## Usage

### Query Endpoint

Send questions to the agent via `POST /query`:

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "list sales orders"}'
```

**Response (success):**
```json
{
  "answer": "Here are the sales orders: SO-001 ($100), SO-002 ($250), ..."
}
```

**Response (error):**

When `tool_result` contains `{"error": "..."}`, or the MCP call fails, `POST /query` returns that failure directly:

```json
{
  "error": "Employee not found"
}
```

```json
{
  "error": "MCP server unreachable"
}
```

**Example questions:**
- `"hello"` — Direct greeting (no MCP call)
- `"list sales orders"` — Calls `list_sales_order` tool
- `"list employees"` — Calls `list_employees` tool
- `"get employee emp-001"` — Calls `get_employee_by_id` tool (ID normalized to `EMP-001`)
- `"get employee emp001"` — Calls `get_employee_by_id` tool (ID normalized to `EMP001`)
- `"get sales order so-123"` — Calls `get_sales_order_by_id` tool (ID normalized to `SO-123`)
- `"get sales order so123"` — Calls `get_sales_order_by_id` tool (ID normalized to `SO123`)
- `"get sales order ord-001"` — Calls `get_sales_order_by_id` tool (ID normalized to `ORD-001`)

**ID Format:**

The agent normalizes employee and sales-order IDs before calling MCP tools. `agent/mcp_client.py` (`_normalize_id`) follows the previous agent's single rule: if the argument key is `id` and the value is a string, send `value.upper()`. There is no format regex. Separators are kept; only case changes.

Prefix-digits with a hyphen (for example `emp-001`, `so-123`) is the usual form. The same prefixes without a hyphen are supported too. Unhyphenated ids are uppercased (`emp001` → `EMP001`, `so123` → `SO123`). A hyphen-only pattern would leave `emp001` unchanged and drop that support, so it is not used. The automated examples are `emp-001` → `EMP-001` and `ord-001` → `ORD-001`.

- `emp-001`, `Emp-001`, `eMp-001`, `EMP-001` → `EMP-001`
- `emp001`, `Emp001`, `EMP001` → `EMP001`
- `so-123`, `SO-123` → `SO-123`
- `so123` → `SO123`
- `ord-001`, `Ord-001` → `ORD-001`
- `foo-bar-123`, `Foo-Bar-123` → `FOO-BAR-123`
- `emp_001` → `EMP_001`

Other argument keys are left unchanged. Non-string `id` values are left unchanged. The previous tool schema only supplied strings.

## Health Check

Check agent and MCP server status:

```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "up",
  "mcp_reachable": true
}
```

- `status: "up"` — Agent process is running
- `mcp_reachable: true` — MCP server at `MCP_URL` is reachable
- `mcp_reachable: false` — MCP server is down or `MCP_URL` is not set

## Development

### Running Tests

```bash
pytest tests/ -v
```

**Test coverage:**
- Direct LLM answers (no MCP call)
- MCP tool calls over HTTP (all four tools: `list_sales_order`, `list_employees`, `get_sales_order_by_id`, `get_employee_by_id`)
- ID normalization (any string `id` is uppercased, including prefix-digits with a hyphen such as `emp-001` and unhyphenated forms such as `emp001`)
- Error handling (MCP unreachable, tool errors returned as `{"error": "..."}`, LLM unavailable)
- Health endpoint

### Project Structure

```
adlc-agents/
├── agent/
│   ├── main.py            # FastAPI app entry point
│   ├── orchestrator.py    # Routes questions to LLM/MCP
│   ├── llm_client.py      # LLM client wrapper
│   ├── mcp_client.py      # FastMCP HTTP client
│   └── health.py          # Health check logic
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
