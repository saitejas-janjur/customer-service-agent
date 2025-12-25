"""
LangChain tool adapters.

Purpose:
- Convert our production ToolExecutor (Phase 2) into LangChain Tools.
- Add a KB retrieval tool powered by Phase 1 HybridRetriever.

IMPORTANT FIX:
LangChain may pass tool inputs as JSON strings instead of dicts.
We defensively parse inputs before sending them to ToolExecutor.
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


def _ensure_dict(value: Any) -> dict[str, Any]:
    """
    Ensure tool arguments are a dict.

    LangChain ReAct may pass:
    - dict ✅
    - JSON string ❌

    This function normalizes both into a dict.
    """
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Tool input must be a dict or JSON object string: {value}")


def _format_kb_results(hits: list[RetrievedChunk]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        citation = str(h.doc.metadata.get("citation", "unknown"))
        snippet = (h.doc.page_content or "").strip()[:900]
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
    tools: list[StructuredTool] = []
    tools.extend(_build_business_tools(executor=executor, ctx=ctx))
    if enable_kb_tool:
        tools.append(_build_kb_tool(ctx=ctx))
    return tools


def _build_business_tools(
    *, executor: ToolExecutor, ctx: ToolContext
) -> list[StructuredTool]:

    async def run_tool(name: str, raw_args: Any) -> str:
        try:
            args = _ensure_dict(raw_args)
            out = await executor.execute(ctx, ToolCall(name=name, args=args))
            return json.dumps(out.model_dump(), ensure_ascii=False)
        except ToolError as e:
            return json.dumps(
                {"error": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            )
        except Exception as e:
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

    from app.tools.schemas import (
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
            description="Issue a refund (policy enforced server-side).",
            args_schema=IssueRefundInput,
            coroutine=issue_refund,
        ),
        StructuredTool.from_function(
            name="update_contact",
            description="Update the authenticated user's contact info.",
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

    async def search_knowledge_base(query: str, k: int = 5) -> str:
        try:
            from app.config import get_settings

            s = get_settings()
            retriever = load_hybrid_retriever(s)
            hits = retriever.retrieve(query)[:k]
            return json.dumps(_format_kb_results(hits), ensure_ascii=False)
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
            "Search the internal policy/FAQ knowledge base and return numbered "
            "snippets with citations."
        ),
        args_schema=SearchKBInput,
        coroutine=search_knowledge_base,
    )