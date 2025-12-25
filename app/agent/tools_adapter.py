"""
LangChain tool adapters.

Problem we solve:
- LangChain's default ReAct parser provides `Action Input:` as a STRING.
- We must parse that string into a dict (JSON) before calling our Phase 2
  ToolExecutor (which performs server-side validation and policy checks).

Also:
- Some LangChain versions require constructing tools via Tool.from_function()
  (Tool(...) constructor may require a positional func).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from langchain_core.tools import BaseTool, Tool

from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.kb_loader import load_hybrid_retriever
from app.tools.executor import ToolCall, ToolExecutor
from app.tools.types import ToolContext, ToolError


def _format_kb_results(hits: list[RetrievedChunk]) -> dict[str, Any]:
    """
    Format KB hits into a tool-friendly structure.

    The agent should cite snippet numbers like [1], [2] in the final answer.
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
) -> list[BaseTool]:
    """
    Build the list of tools exposed to the ReAct agent.
    """
    tools: list[BaseTool] = []
    tools.extend(_build_business_tools(executor=executor, ctx=ctx))

    if enable_kb_tool:
        tools.append(_build_kb_tool())

    return tools


def _try_parse_json_object(tool_input: str) -> dict[str, Any] | None:
    """
    Parse tool_input if it looks like a JSON object. Return None if not parseable.
    """
    s = (tool_input or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


def _heuristic_args(tool_name: str, tool_input: str) -> dict[str, Any] | None:
    """
    Heuristics for non-JSON tool inputs.

    Only allow heuristics where it is unambiguous and safe.
    Otherwise require JSON.
    """
    s = (tool_input or "").strip()

    if tool_name == "get_order_status":
        # Allow: Action Input: ord_XXXX
        return {"order_id": s}

    if tool_name == "track_shipment":
        # Allow: Action Input: ord_... OR trk_...
        if s.startswith("ord_"):
            return {"order_id": s}
        if s.startswith("trk_"):
            return {"tracking_id": s}
        return None

    if tool_name == "initiate_password_reset":
        # Allow: Action Input: customer@example.com
        return {"email": s}

    # For refund + update_contact: require JSON to avoid ambiguity.
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

    Returns:
        A JSON string observation for the agent.
    """
    args = _try_parse_json_object(tool_input)
    if args is None:
        args = _heuristic_args(tool_name, tool_input)

    if args is None:
        return json.dumps(
            {
                "error": "ToolInputError",
                "message": (
                    "Tool input must be a JSON object string for this tool. "
                    "Example: {\"field\": \"value\"}"
                ),
                "tool": tool_name,
                "received": tool_input,
            },
            ensure_ascii=False,
        )

    try:
        out = await executor.execute(ctx, ToolCall(name=tool_name, args=args))
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


def _make_async_tool(
    *,
    name: str,
    description: str,
    coroutine: Callable[[str], Awaitable[str]],
) -> BaseTool:
    """
    Create a Tool in a version-compatible way.

    Some LangChain versions require `func` for Tool construction, even if we only
    use async execution. We provide a sync stub to satisfy the signature.
    """

    def _sync_stub(tool_input: str) -> str:
        raise RuntimeError(
            f"Tool '{name}' is async-only in this project. Use arun/ainvoke."
        )

    return Tool.from_function(
        name=name,
        description=description,
        func=_sync_stub,
        coroutine=coroutine,
    )


def _build_business_tools(*, executor: ToolExecutor, ctx: ToolContext) -> list[BaseTool]:
    """
    Create LangChain tools for Phase 2 business operations.

    ReAct calls tools with a single string argument, so each tool signature is:
      tool_input: str
    """

    async def get_order_status(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor,
            ctx=ctx,
            tool_name="get_order_status",
            tool_input=tool_input,
        )

    async def track_shipment(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor,
            ctx=ctx,
            tool_name="track_shipment",
            tool_input=tool_input,
        )

    async def issue_refund(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor,
            ctx=ctx,
            tool_name="issue_refund",
            tool_input=tool_input,
        )

    async def update_contact(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor,
            ctx=ctx,
            tool_name="update_contact",
            tool_input=tool_input,
        )

    async def initiate_password_reset(tool_input: str) -> str:
        return await _run_business_tool(
            executor=executor,
            ctx=ctx,
            tool_name="initiate_password_reset",
            tool_input=tool_input,
        )

    return [
        _make_async_tool(
            name="get_order_status",
            description=(
                "Get order status/details. Input: JSON string like "
                "{\"order_id\":\"ord_...\"} OR plain order id 'ord_...'."
            ),
            coroutine=get_order_status,
        ),
        _make_async_tool(
            name="track_shipment",
            description=(
                "Track shipment. Input: JSON string like {\"order_id\":\"ord_...\"} "
                "or {\"tracking_id\":\"trk_...\"}, OR plain id 'ord_...'/'trk_...'."
            ),
            coroutine=track_shipment,
        ),
        _make_async_tool(
            name="issue_refund",
            description=(
                "Issue a refund (policy enforced). Input MUST be JSON string: "
                "{\"order_id\":\"ord_...\",\"amount_usd\":50,"
                "\"reason\":\"Damaged item\",\"idempotency_key\":\"...\"}"
            ),
            coroutine=issue_refund,
        ),
        _make_async_tool(
            name="update_contact",
            description=(
                "Update contact info. Input MUST be JSON string: "
                "{\"new_email\":\"...\"} and/or {\"new_phone_e164\":\"+1415...\"}"
            ),
            coroutine=update_contact,
        ),
        _make_async_tool(
            name="initiate_password_reset",
            description=(
                "Initiate password reset. Input: JSON string {\"email\":\"...\"} "
                "OR plain email address."
            ),
            coroutine=initiate_password_reset,
        ),
    ]


def _build_kb_tool() -> BaseTool:
    """
    KB search tool for policy/FAQ.

    Input:
    - Either plain text query (recommended)
    - Or JSON string: {"query":"...", "k": 5}
    """

    async def search_knowledge_base(tool_input: str) -> str:
        try:
            from app.config import get_settings

            s = get_settings()

            args = _try_parse_json_object(tool_input)
            if args is None:
                query = tool_input.strip()
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
                {
                    "error": type(e).__name__,
                    "message": str(e),
                    "hint": "Ensure KB exists: python scripts/build_kb.py",
                },
                ensure_ascii=False,
            )

    return _make_async_tool(
        name="search_knowledge_base",
        description=(
            "Search policy/FAQ knowledge base. Input: plain query string, or JSON "
            "string {\"query\":\"...\",\"k\":5}. Returns numbered snippets to cite."
        ),
        coroutine=search_knowledge_base,
    )