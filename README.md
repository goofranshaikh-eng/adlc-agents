# adlc-agents

FastAPI service that accepts an agent query and returns an `agent_run_id` immediately.

## POST /agent/query

Submit a user query. The handler validates the body, records an in-memory run, and responds before any LLM or MCP work finishes. `agent_run_id` is a UUID v4 hex string (32 lowercase characters). The same id identifies the run and the event queue a later SSE subscriber reads.

Request:

```json
{"query": "What is the weather?"}
```

Accepted (HTTP 200):

```json
{"status": "ok", "agent_run_id": "<uuid v4 hex>"}
```

Rejected when `query` is missing, not a string, empty, or whitespace-only (HTTP 200, not 422):

```json
{"status": "rejected", "agent_run_id": "<uuid v4 hex>", "reason": "invalid query"}
```

On rejection the run is stored with an `in_progress` event followed by a `failed` event (`error` is `invalid query`). No background job is started. On acceptance the run stays `pending` and a background task is scheduled without being awaited.

## Other routes

- `GET /health` returns `{"status": "ok"}`.
- `POST /query` is the existing query endpoint and is unchanged by this story.

## Run

```bash
pip install -r requirements.txt
uvicorn agent.main:app --reload
```

## Tests

```bash
python -m pytest agent/tests -q
```
