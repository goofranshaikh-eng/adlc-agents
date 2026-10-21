from fastapi import FastAPI, Request

from agent.health import check_health

app = FastAPI()


@app.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question", "")
    # Stub: return placeholder until LLM integrated (Story 2)
    return {"answer": "Placeholder response. LLM integration pending Story 2."}


@app.get("/health")
async def health():
    return await check_health()
