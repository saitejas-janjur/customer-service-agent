"""
Tool Execution Node.

Responsibilities:
- Inspect the last message for 'tool_calls'.
- Map them to our internal ToolExecutor.
- Run them safely (with Phase 2 auditing/policies).
- Return result as a ToolMessage.
"""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from app.config import Settings
from app.graph.state import AgentState
from app.tools.audit import ToolAuditLogger
from app.tools.executor import ToolCall, ToolExecutor
from app.tools.factory import build_tool_registry
from app.tools.types import ToolContext


async def tools_node(state: AgentState, settings: Settings) -> dict:
    last_msg = state["messages"][-1]
    
    # Should not happen due to routing, but safe guard:
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}

    registry, _store = build_tool_registry(settings)
    audit = ToolAuditLogger(audit_dir=settings.audit_dir)
    executor = ToolExecutor(registry=registry, settings=settings, audit_logger=audit)

    ctx = ToolContext(
        user_id=state["user_id"],
        request_id=state["request_id"],
        actor="customer"
    )

    results = []

    for call in last_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_call_id = call["id"]

        # Execute our internal Safe Tool
        # We catch all exceptions here to report them back to the LLM
        # rather than crashing the graph.
        try:
            internal_call = ToolCall(name=tool_name, args=tool_args)
            output_model = await executor.execute(ctx, internal_call)
            
            # Serialize for LLM
            content = output_model.model_dump_json()
        except Exception as e:
            content = json.dumps(
                {"error": type(e).__name__, "message": str(e)},
                ensure_ascii=False
            )

        results.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=content
            )
        )

    return {"messages": results}