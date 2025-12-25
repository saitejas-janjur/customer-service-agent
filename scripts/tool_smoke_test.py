"""
Phase 2 smoke test.

Run:
  python scripts/tool_smoke_test.py

Expected:
- Tool list
- Order status + shipment tracking results
- Refund policy enforcement (cap/window)
- Contact update and password reset flow
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app.config import get_settings
from app.tools.executor import ToolCall, ToolExecutor
from app.tools.factory import build_tool_registry
from app.tools.types import RequestId, ToolContext, UserId
from app.utils.logging import setup_logging


async def main() -> None:
    s = get_settings()
    setup_logging(s.log_level)

    registry, _store = build_tool_registry(s)
    executor = ToolExecutor(registry=registry, settings=s)

    ctx = ToolContext(
        user_id=UserId("user_123"),
        request_id=RequestId(f"req_{uuid4().hex[:12]}"),
        actor="customer",
    )

    print("Registered tools:", registry.names())
    print("")

    # 1) Get order status
    out1 = await executor.execute(
        ctx,
        ToolCall(name="get_order_status", args={"order_id": "ord_XYZ78901"}),
    )
    print("get_order_status:", out1.model_dump())
    print("")

    # 2) Track shipment (via order_id)
    out2 = await executor.execute(
        ctx,
        ToolCall(name="track_shipment", args={"order_id": "ord_XYZ78901"}),
    )
    print("track_shipment:", out2.model_dump())
    print("")

    # 3) Try refund above policy cap (should fail)
    try:
        await executor.execute(
            ctx,
            ToolCall(
                name="issue_refund",
                args={
                    "order_id": "ord_XYZ78901",
                    "amount_usd": 150.0,
                    "reason": "Item arrived damaged",
                },
            ),
        )
    except Exception as e:
        print("issue_refund (expected failure):", type(e).__name__, str(e))
    print("")

    # 4) Refund within cap (should pass)
    out4 = await executor.execute(
        ctx,
        ToolCall(
            name="issue_refund",
            args={
                "order_id": "ord_XYZ78901",
                "amount_usd": 50.0,
                "reason": "Item arrived damaged",
                "idempotency_key": "refund_once_001",
            },
        ),
    )
    print("issue_refund:", out4.model_dump())
    print("")

    # 5) Update contact
    out5 = await executor.execute(
        ctx,
        ToolCall(
            name="update_contact",
            args={"new_email": "customer+new@example.com"},
        ),
    )
    print("update_contact:", out5.model_dump())
    print("")

    # 6) Initiate password reset
    out6 = await executor.execute(
        ctx,
        ToolCall(
            name="initiate_password_reset",
            args={"email": "customer+new@example.com"},
        ),
    )
    print("initiate_password_reset:", out6.model_dump())
    print("")


if __name__ == "__main__":
    asyncio.run(main())