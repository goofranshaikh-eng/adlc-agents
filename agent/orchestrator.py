from agent import llm_client, mcp_client


async def process_question(question: str) -> dict:
    try:
        llm_response = await llm_client.ask_llm(question)

        if llm_response["type"] == "direct_answer":
            return {"answer": llm_response["answer"]}

        if llm_response["type"] == "tool_request":
            tool_name = llm_response["tool_name"]
            tool_args = llm_response["tool_args"]

            tool_result = await mcp_client.call_tool(tool_name, tool_args)

            synthesized = await llm_client.ask_llm(question, tool_result=tool_result)
            return {"answer": synthesized["answer"]}

        return {"error": "LLM unavailable"}
    except Exception:
        return {"error": "LLM unavailable"}
