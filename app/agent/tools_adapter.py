"""
LangChain tool adapters.

Purpose:
- Convert our production ToolExecutor (Phase 2) into LangChain Tools.
- Add a KB retrieval tool powered by Phase 1 HybridRetriever.

We create tools PER REQUEST because they need:
- ToolContext (user_id, request_id)
- ToolExecutor with audit logging
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.kb_loader import load_hybrid_retriever
from app.tools.executor import ToolCall, ToolExecutor
from app.tools.types import ToolContext, ToolError


class SearchKBInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=8)


def _format_kb_results(hits: list[RetrievedChunk]) -> dict[str, Any]:
    """
    Format KB hits into a tool-friendly structure.

    The agent prompt asks the user-facing answer to cite [1], [2], ...
    We therefore return a numbered list of snippets with citations.
    """
    items: list[dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        citation = str(h.doc.metadata.get("citation", "unknown"))
        snippet = (h.doc.page_content or "").strip()
        snippet = snippet[:900]
        items.append(
            {
                "n": i,
                "citation": citation,
                "snippet": snippet,
                "score": round(float(h.score), 4),
            }
        )

    return {"results": items}


def build_langchain_tools(
    *,
    executor: ToolExecutor,
    ctx: ToolContext,
    enable_kb_tool: bool = True,
) -> list[StructuredTool]:
    """
    Build the tool list exposed to the agent.

    Tools included:
    - Phase 2 business tools (order/shipment/refund/contact/password reset)
    - Optional: KB search tool for policy questions
    """
    tools: list[StructuredTool] = []

    tools.extend(_build_business_tools(executor=executor, ctx=ctx))

    if enable_kb_tool:
        tools.append(_build_kb_tool(ctx=ctx))

    return tools


def _build_business_tools(*, executor: ToolExecutor, ctx: ToolContext) -> list[StructuredTool]:
    """
    Wrap each Phase 2 tool in a LangChain StructuredTool.
    """

    async def run_tool(name: str, args: dict[str, Any]) -> str:
        try:
            out = await executor.execute(ctx, ToolCall(name=name, args=args))
            return json.dumps(out.model_dump(), ensure_ascii=False)
        except ToolError as e:
            # Return an observation string the agent can react to.
            return json.dumps(
                {"error": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            )

    async def get_order_status(**kwargs: Any) -> str:
        return await run_tool("get_order_status", kwargs)

    async def track_shipment(**kwargs: Any) -> str:
        return await run_tool("track_shipment", kwargs)

    async def issue_refund(**kwargs: Any) -> str:
        return await run_tool("issue_refund", kwargs)

    async def update_contact(**kwargs: Any) -> str:
        return await run_tool("update_contact", kwargs)

    async def initiate_password_reset(**kwargs: Any) -> str:
        return await run_tool("initiate_password_reset", kwargs)

    # Note: args_schema links to our Pydantic tool inputs from Phase 2.
    from app.tools.schemas import (  # local import keeps module load light
        GetOrderStatusInput,
        InitiatePasswordResetInput,
        IssueRefundInput,
        TrackShipmentInput,
        UpdateContactInput,
    )

    return [
        StructuredTool.from_function(
            name="get_order_status",
            description="Get the status and details of an order by order_id.",
            args_schema=GetOrderStatusInput,
            coroutine=get_order_status,
        ),
        StructuredTool.from_function(
            name="track_shipment",
            description="Track a shipment by tracking_id OR by order_id.",
            args_schema=TrackShipmentInput,
            coroutine=track_shipment,
        ),
        StructuredTool.from_function(
            name="issue_refund",
            description=(
                "Issue a refund for an order. Enforces refund policy caps and window."
            ),
            args_schema=IssueRefundInput,
            coroutine=issue_refund,
        ),
        StructuredTool.from_function(
            name="update_contact",
            description="Update the authenticated user's email and/or phone number.",
            args_schema=UpdateContactInput,
            coroutine=update_contact,
        ),
        StructuredTool.from_function(
            name="initiate_password_reset",
            description="Initiate password reset for the authenticated user's email.",
            args_schema=InitiatePasswordResetInput,
            coroutine=initiate_password_reset,
        ),
    ]


def _build_kb_tool(*, ctx: ToolContext) -> StructuredTool:
    """
    KB tool is not a Phase 2 business tool; it's a retrieval tool for FAQs/policies.

    It loads the on-disk KB and retrieves top-k snippets.
    """

    async def search_knowledge_base(query: str, k: int = 5) -> str:
        try:
            # Load retriever lazily per call (simple + reliable).
            # If you want higher performance later, we can cache this.
            from app.config import get_settings

            s = get_settings()
            retriever = load_hybrid_retriever(s)
            hits = retriever.retrieve(query)[:k]
            payload = _format_kb_results(hits)
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {
                    "error": type(e).__name__,
                    "message": str(e),
                    "hint": "Ensure KB exists: python scripts/build_kb.py",
                },
                ensure_ascii=False,
            )

    return StructuredTool.from_function(
        name="search_knowledge_base",
        description=(
            "Search the internal policy/FAQ knowledge base and return numbered snippets "
            "with citations. Use for policy questions like refunds, shipping, returns."
        ),
        args_schema=SearchKBInput,
        coroutine=search_knowledge_base,
    )