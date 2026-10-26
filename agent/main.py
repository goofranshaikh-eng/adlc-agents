"""FastAPI application: immediate accept for POST /agent/query."""

import asyncio

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agent.models import AgentQueryAccepted, AgentQueryRejected, AgentQueryRequest
from agent.run_store import RunStore

app = FastAPI()

# Global singleton run store
run_store = RunStore()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors on POST /agent/query to HTTP 200 rejected.

    FastAPI raises RequestValidationError (HTTP 422) when query is missing or is
    not a string. This handler runs before the route body and returns HTTP 200
    with status rejected.

    Other routes keep FastAPI's default 422 response.
    """
    if request.method != "POST" or request.url.path != "/agent/query":
        return await request_validation_exception_handler(request, exc)

    run = await run_store.create("")
    await run_store.mark_failed(run.id, "invalid query")
    return JSONResponse(
        status_code=200,
        content={
            "status": "rejected",
            "agent_run_id": run.id,
            "reason": "invalid query",
        },
    )


@app.post("/agent/query", response_model=AgentQueryAccepted | AgentQueryRejected)
async def post_agent_query(
    req: AgentQueryRequest,
) -> AgentQueryAccepted | AgentQueryRejected:
    """Accept a user query immediately and return agent_run_id.

    AC-1: Return HTTP 200 immediately without waiting for process_question.
    AC-2: Reject an empty or whitespace-only query and do not spawn a background job.
    """
    query = req.query.strip()

    if not query:
        run = await run_store.create("")
        await run_store.mark_failed(run.id, "invalid query")
        return AgentQueryRejected(
            status="rejected",
            agent_run_id=run.id,
            reason="invalid query",
        )

    run = await run_store.create(query)

    # Do not await. The handler returns before LLM or MCP work starts.
    asyncio.create_task(run_background_job(run.id, query))

    return AgentQueryAccepted(status="ok", agent_run_id=run.id)


async def run_background_job(agent_run_id: str, query: str) -> None:
    """Placeholder background orchestration spawned via asyncio.create_task.

    Story 1 performs no work here. Story 2 calls process_question, applies the
    5-second hold, and enqueues completed or failed.

    Args:
        agent_run_id: unique run identifier (UUID v4 hex)
        query: user query string already validated and stripped
    """
    return None


@app.post("/query", response_model=None)
async def post_query(question: str) -> dict:
    """Existing blocking query endpoint (PER-20).

    Unchanged by this story. Story 3 verifies backward compatibility.
    Story 1 does not call process_question. response_model is disabled so this
    placeholder does not fail FastAPI response validation.
    """
    pass


@app.get("/health")
async def get_health() -> dict:
    """Existing health check endpoint."""
    return {"status": "ok"}
