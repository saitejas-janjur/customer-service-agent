"""
Agent (Reasoning) Node.

The core LLM loop:
1. Constructs a system prompt with context (if any).
2. Binds available tools.
3. Generates the next message (text answer or tool_call).
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agent.tools_adapter import build_langchain_tools
from app.config import Settings
from app.graph.state import AgentState
from app.llm.models import build_reasoning_llm
from app.tools.executor import ToolExecutor
from app.tools.factory import build_tool_registry
from app.tools.types import ToolContext


async def agent_node(state: AgentState, settings: Settings) -> dict:
    # 1. Setup Context & Tools
    registry, _store = build_tool_registry(settings)
    # Note: We create a dummy logger/executor here just to get the tool definitions.
    # The actual execution happens in the 'tools_node'.
    executor = ToolExecutor(registry=registry, settings=settings)
    
    ctx = ToolContext(
        user_id=state["user_id"],
        request_id=state["request_id"],
        actor="customer"
    )
    
    # We intentionally exclude the 'search_knowledge_base' tool here because
    # retrieval is handled by the explicit retrieval_node upstream.
    tools = build_langchain_tools(executor=executor, ctx=ctx, enable_kb_tool=False)
    
    # 2. Build Prompt
    # We inject retrieved docs into the system prompt if they exist.
    docs = state.get("retrieved_docs", [])
    context_str = ""
    if docs:
        snippets = [
            f"[{i}] {d.doc.page_content[:400]}... (Source: {d.doc.metadata.get('citation')})"
            for i, d in enumerate(docs, 1)
        ]
        context_str = "\nREFERENCE MATERIAL:\n" + "\n\n".join(snippets)

    system_msg = (
        "You are a helpful customer service assistant for an e-commerce store.\n"
        "Use the provided tools to assist with orders, refunds, and tracking.\n"
        "If you have Reference Material, use it to answer questions. Cite sources like [1].\n"
        "If you cannot help, polite decline."
        f"{context_str}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 3. Call Model
    llm = build_reasoning_llm(settings)
    llm_with_tools = llm.bind_tools(tools)
    
    chain = prompt | llm_with_tools
    
    response = await chain.ainvoke({"messages": state["messages"]})
    
    return {"messages": [response]}