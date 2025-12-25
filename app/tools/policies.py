"""
Business policies for tools.

Key rule:
- Policies are enforced in code, not prompts.
- If a policy is violated, raise ToolPolicyError (not ToolValidationError).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.config import Settings
from app.tools.types import ToolPolicyError
from app.tools.store import Order


def assert_order_belongs_to_user(order: Order, user_id: str) -> None:
    if order.user_id != user_id:
        raise ToolPolicyError("Order does not belong to the current user.")


def assert_refund_within_window(order: Order, settings: Settings) -> None:
    now = datetime.now(UTC)
    window = timedelta(days=int(settings.refund_window_days))
    if now - order.created_at > window:
        raise ToolPolicyError(
            f"Refund window expired ({settings.refund_window_days} days)."
        )


def assert_refund_amount_allowed(
    order: Order,
    *,
    amount_usd: float,
    settings: Settings,
) -> None:
    cap = float(settings.refund_max_amount_usd)
    if amount_usd > cap:
        raise ToolPolicyError(
            f"Refund amount exceeds policy cap of ${cap:.2f}."
        )

    remaining = max(0.0, float(order.total_amount_usd) - float(order.refunded_amount_usd))
    if amount_usd > remaining + 1e-9:
        raise ToolPolicyError(
            f"Refund exceeds remaining refundable amount (${remaining:.2f})."
        )

    if order.status == "cancelled":
        raise ToolPolicyError("Cannot refund a cancelled order via this tool.")