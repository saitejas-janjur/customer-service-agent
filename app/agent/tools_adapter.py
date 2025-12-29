"""
LangChain tool adapters.

Updates in this version:
- FIXED: Use `out.model_dump_json()` instead of `json.dumps(out.model_dump())`.
  This ensures `datetime` fields from Pydantic models are serialized correctly
  (preventing "Object of type datetime is not JSON serializable" errors).
- Includes robust input parsing (stripping quotes/markdown).
- Uses `Tool.from_function` for compatibility.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable

from langchain_core.tools import BaseTool, Tool

from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.kb_loader import load_hybrid_retriever
from app.tools.executor import ToolCall, ToolExecutor
from app.tools.types import ToolContext, ToolError


def _format_kb_results(hits: list[RetrievedChunk]) -> dict[str, Any]:
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
) -> list[BaseTool]:
    tools: list[BaseTool] = []
    tools.extend(_build_business_tools(executor=executor, ctx=ctx))

    if enable_kb_tool:
        tools.append(_build_kb_tool())

    return tools


def _clean_input(text: str) -> str:
    """
    Clean LLM output artifacts from the tool input string.
    - Removes markdown code blocks (```json ... ```)
    - Removes surrounding quotes ("...", '...')
    """
    s = (text or "").strip()

    # Remove markdown code blocks
    if "```" in s:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, s, re.DOTALL)
        if match:
            s = match.group(1).strip()

    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    return s


def _try_parse_json_object(tool_input: str) -> dict[str, Any] | None:
    """
    Parse tool_input if it looks like a JSON object. Return None if not parseable.
    """
    s = _clean_input(tool_input)
    
    if not (s.startswith("{") and s.endswith("}")):
        return None
    
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


def _heuristic_args(tool_name: str, tool_input: str) -> dict[str, Any] | None:
    """
    Heuristics for non-JSON tool inputs.
    """
    s = _clean_input(tool_input)

    if tool_name == "get_order_status":
        # Allow: ord_XXXX
        return {"order_id": s}

    if tool_name == "track_shipment":
        # Allow: ord_... OR trk_...
        if s.startswith("ord_"):
            return {"order_id": s}
        if s.startswith("trk_"):
            return {"tracking_id": s}
        return None

    if tool_name == "initiate_password_reset":
        # Allow: customer@example.com
        return {"email": s}
    
    if tool_name == "search_knowledge_base":
        return {"query": s, "k": 5}

    return None


async def _run_business_tool(
    *,
    executor: ToolExecutor,
    ctx: ToolContext,
    tool_name: str,
    tool_input: str,
) -> str:
    """
    Execute a Phase 2 business tool via our ToolExecutor.
    Returns: JSON string observation.
    """
    args = _try_parse_json_object(tool_input)
    if args is None:
        args = _heuristic_args(tool_name, tool_input)

    if args is None:
        return json.dumps(
            {
                "error": "ToolInputError",
                "message": (
                    f"Could not parse input for tool '{tool_name}'. "
                    "Please provide a valid JSON string."
                ),
                "received_raw": tool_input,
            },
            ensure_ascii=False,
        )

    try:
        out = await executor.execute(ctx, ToolCall(name=tool_name, args=args))
        # --- CRITICAL FIX HERE ---
        # Use .model_dump_json() to handle datetime serialization automatically.
        return out.model_dump_json()
        # -------------------------
    except ToolError as e:
        return json.dumps(
            {"error": type(e).__name__, "message": str(e)},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"error": "UnexpectedError", "message": str(e)},
            ensure_ascii=False,
        )


def _make_async_tool(
    *,
    name: str,
    description: str,
    coroutine: Callable[[str], Awaitable[str]],
) -> BaseTool:
    def _sync_stub(tool_input: str) -> str:
        raise RuntimeError(f"Tool '{name}' is async-only. Use arun/ainvoke.")

    return Tool.from_function(
        name=name,
        description=description,
        func=_sync_stub,
        coroutine=coroutine,
    )


def _build_business_tools(*, executor: ToolExecutor, ctx: ToolContext) -> list[BaseTool]:
    
    async def get_order_status(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor, ctx=ctx, tool_name="get_order_status", tool_input=tool_input
        )

    async def track_shipment(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor, ctx=ctx, tool_name="track_shipment", tool_input=tool_input
        )

    async def issue_refund(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor, ctx=ctx, tool_name="issue_refund", tool_input=tool_input
        )

    async def update_contact(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor, ctx=ctx, tool_name="update_contact", tool_input=tool_input
        )

    async def initiate_password_reset(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor, ctx=ctx, tool_name="initiate_password_reset", tool_input=tool_input
        )

    return [
        _make_async_tool(
            name="get_order_status",
            description="Get order status. Input: JSON string or plain order ID.",
            coroutine=get_order_status,
        ),
        _make_async_tool(
            name="track_shipment",
            description="Track shipment. Input: JSON string or plain order/tracking ID.",
            coroutine=track_shipment,
        ),
        _make_async_tool(
            name="issue_refund",
            description="Issue refund. Input: JSON string.",
            coroutine=issue_refund,
        ),
        _make_async_tool(
            name="update_contact",
            description="Update contact. Input: JSON string.",
            coroutine=update_contact,
        ),
        _make_async_tool(
            name="initiate_password_reset",
            description="Reset password. Input: JSON string or plain email.",
            coroutine=initiate_password_reset,
        ),
    ]


def _build_kb_tool() -> BaseTool:
    async def search_knowledge_base(tool_input: str) -> str:
        try:
            from app.config import get_settings
            s = get_settings()

            args = _try_parse_json_object(tool_input)
            if args is None:
                query = _clean_input(tool_input)
                k = 5
            else:
                query = str(args.get("query", "")).strip()
                k = int(args.get("k", 5))

            retriever = load_hybrid_retriever(s)
            hits = retriever.retrieve(query)[: max(1, min(k, 8))]
            payload = _format_kb_results(hits)
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {"error": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            )

    return _make_async_tool(
        name="search_knowledge_base",
        description="Search knowledge base. Input: plain query string or JSON.",
        coroutine=search_knowledge_base,
    )