import asyncio
import os
import re

from anthropic import Anthropic

_client = None

_GET_BY_ID = (
    (
        "get_employee_by_id",
        re.compile(r"\bget\s+employee\s+([A-Za-z]+-\d+)\b", re.IGNORECASE),
    ),
    (
        "get_sales_order_by_id",
        re.compile(r"\bget\s+sales\s+order\s+([A-Za-z]+-\d+)\b", re.IGNORECASE),
    ),
)


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
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _is_greeting_or_help(question: str) -> bool:
    greetings = ["hello", "hi", "help", "what can you do"]
    return any(g in question.lower() for g in greetings)


def _infer_tool(question: str) -> tuple[str, dict]:
    """Map a data question to one of the four MCP tools.

    ``get <entity> <id>`` is checked before the list heuristics so
    ``get employee emp-001`` does not collapse into ``list_employees``
    and drop the id.
    """
    for tool_name, pattern in _GET_BY_ID:
        match = pattern.search(question)
        if match:
            return tool_name, {"id": match.group(1)}

    lowered = question.lower()
    if "sales order" in lowered:
        return "list_sales_order", {}
    if "employee" in lowered:
        return "list_employees", {}
    return "list_sales_order", {}


async def ask_llm(question: str, tool_result: dict | None = None) -> dict:
    client = _get_client()

    if tool_result is not None:
        prompt = (
            f"User asked: {question}\n"
            f"Tool returned: {tool_result}\n"
            "Synthesize a natural language answer."
        )
        response = await asyncio.to_thread(_call_llm_sync, client, prompt)
        return {"type": "direct_answer", "answer": response}

    if _is_greeting_or_help(question):
        response = await asyncio.to_thread(_call_llm_sync, client, question)
        return {"type": "direct_answer", "answer": response}

    tool_name, tool_args = _infer_tool(question)
    return {
        "type": "tool_request",
        "tool_name": tool_name,
        "tool_args": tool_args,
    }
