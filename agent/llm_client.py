import asyncio
import os
import re

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

# Entity-by-id handlers. Checked before list routing so "get employee emp-001"
# cannot collapse into list_employees and drop the id. The captured id is the
# raw question text (emp-001, emp001, Emp-001, so-123, so123, ord-001);
# mcp_client uppercases it before the FastMCP call.
_GET_BY_ID_HANDLERS = (
    (
        "get_employee_by_id",
        re.compile(
            r"\b(?:get\s+employee(?:\s+with\s+id)?|employee\s+with\s+id)"
            r"\s+([A-Za-z][\w-]*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "get_sales_order_by_id",
        re.compile(
            r"\b(?:get\s+sales\s+order(?:\s+with\s+id)?|sales\s+order\s+with\s+id)"
            r"\s+([A-Za-z][\w-]*)\b",
            re.IGNORECASE,
        ),
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
        model=_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _tool_args(block) -> dict:
    raw = getattr(block, "input", None) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _infer_get_by_id(question: str) -> tuple[str, dict] | None:
    """Return ``(tool_name, {"id": raw})`` for an entity-by-id question.

    Handlers: ``get_employee_by_id``, ``get_sales_order_by_id``.
    List questions and greetings return None so the model still chooses those.
    """
    for tool_name, pattern in _GET_BY_ID_HANDLERS:
        match = pattern.search(question)
        if match:
            return tool_name, {"id": match.group(1)}
    return None


def _apply_get_by_id(question: str, choice: dict) -> dict:
    """Keep entity-by-id queries on the matching tool with the extracted id.

    A model that picks a list tool, omits ``id``, or answers in prose would
    drop the record. The handler result wins for those questions. The id is
    the raw token from the question; normalization happens in ``mcp_client``.
    """
    inferred = _infer_get_by_id(question)
    if inferred is None:
        return choice

    tool_name, tool_args = inferred
    if choice.get("type") == "tool_request" and choice.get("tool_name") == tool_name:
        merged = dict(choice.get("tool_args") or {})
        merged["id"] = tool_args["id"]
        return {
            "type": "tool_request",
            "tool_name": tool_name,
            "tool_args": merged,
        }
    return {"type": "tool_request", "tool_name": tool_name, "tool_args": tool_args}


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
            choice = {"type": "direct_answer", "answer": f"Unknown tool {name}"}
        else:
            choice = {
                "type": "tool_request",
                "tool_name": name,
                "tool_args": _tool_args(block),
            }
        return _apply_get_by_id(question, choice)
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    return _apply_get_by_id(
        question, {"type": "direct_answer", "answer": text or "No answer."}
    )


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
