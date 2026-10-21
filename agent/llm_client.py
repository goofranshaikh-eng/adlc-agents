import asyncio
import os

from anthropic import Anthropic

_client = None
_MODEL = "claude-sonnet-4-5-20250929"
_TOOLS = [
    {
        "name": "list_sales_order",
        "description": "List every sales order. No arguments.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_employees",
        "description": "List every employee. No arguments.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_sales_order_by_id",
        "description": (
            "Fetch one sales order by id. Hyphenated and unhyphenated forms "
            "are accepted and uppercased before the call "
            "(so-123, so123, ord-001)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "get_employee_by_id",
        "description": (
            "Fetch one employee by id. Hyphenated and unhyphenated forms "
            "are accepted and uppercased before the call "
            "(emp-001, Emp-001, emp001)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
]
_TOOL_NAMES = {tool["name"] for tool in _TOOLS}


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("LLM credentials not set (ANTHROPIC_API_KEY)")
        _client = Anthropic(api_key=api_key)
    return _client


def _call_llm_sync(client, prompt: str) -> str:
    response = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _tool_args(block) -> dict:
    raw = getattr(block, "input", None) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _choose_sync(client, question: str) -> dict:
    """Ask the model which of the four MCP tools matches the question."""
    response = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        tools=_TOOLS,
        messages=[{"role": "user", "content": question}],
    )
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        name = block.name
        if name not in _TOOL_NAMES:
            return {"type": "direct_answer", "answer": f"Unknown tool {name}"}
        return {
            "type": "tool_request",
            "tool_name": name,
            "tool_args": _tool_args(block),
        }
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    return {"type": "direct_answer", "answer": text or "No answer."}


async def ask_llm(question: str, tool_result: dict | None = None) -> dict:
    client = _get_client()

    if tool_result is not None:
        prompt = (
            f"User asked: {question}\n"
            f"Tool returned: {tool_result}\n"
            "Answer from the tool payload only. Do not invent records."
        )
        response = await asyncio.to_thread(_call_llm_sync, client, prompt)
        return {"type": "direct_answer", "answer": response}

    return await asyncio.to_thread(_choose_sync, client, question)
