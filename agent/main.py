from fastapi import FastAPI, Request

from agent.health import check_health
from agent.orchestrator import process_question

app = FastAPI()


@app.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question", "")
    result = await process_question(question)
    return result


@app.get("/health")
async def health():
    return await check_health()
