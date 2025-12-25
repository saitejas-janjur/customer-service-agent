"""
Tool implementations.

Each tool:
- receives ToolContext (who is calling)
- receives validated tool input (Pydantic model)
- performs server-side policy checks
- returns a typed output model

These tools are intentionally "backend-like": no LLM logic inside them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.config import Settings
from app.tools.policies import (
    assert_order_belongs_to_user,
    assert_refund_amount_allowed,
    assert_refund_within_window,
)
from app.tools.registry import Tool
from app.tools.schemas import (
    GetOrderStatusInput,
    GetOrderStatusOutput,
    InitiatePasswordResetInput,
    InitiatePasswordResetOutput,
    IssueRefundInput,
    IssueRefundOutput,
    OrderLineItem,
    TrackShipmentInput,
    TrackShipmentOutput,
    UpdateContactInput,
    UpdateContactOutput,
)
from app.tools.store import InMemoryStore
from app.tools.types import ToolContext, ToolPolicyError, ToolValidationError


class GetOrderStatusTool(Tool[GetOrderStatusInput, GetOrderStatusOutput]):
    name = "get_order_status"
    input_model = GetOrderStatusInput
    output_model = GetOrderStatusOutput

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def run(
        self, ctx: ToolContext, tool_input: GetOrderStatusInput
    ) -> GetOrderStatusOutput:
        order = self._store.get_order(tool_input.order_id)
        if not order:
            raise ToolValidationError("Order not found.")

        assert_order_belongs_to_user(order, ctx.user_id)

        return GetOrderStatusOutput(
            order_id=order.order_id,
            status=order.status,
            created_at=order.created_at,
            total_amount_usd=order.total_amount_usd,
            items=[
                OrderLineItem(
                    sku=i.sku,
                    name=i.name,
                    quantity=i.quantity,
                    unit_price_usd=i.unit_price_usd,
                )
                for i in order.items
            ],
        )


class TrackShipmentTool(Tool[TrackShipmentInput, TrackShipmentOutput]):
    name = "track_shipment"
    input_model = TrackShipmentInput
    output_model = TrackShipmentOutput

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def run(
        self, ctx: ToolContext, tool_input: TrackShipmentInput
    ) -> TrackShipmentOutput:
        tracking_id = tool_input.tracking_id

        if not tracking_id and tool_input.order_id:
            order = self._store.get_order(tool_input.order_id)
            if not order:
                raise ToolValidationError("Order not found.")
            assert_order_belongs_to_user(order, ctx.user_id)
            tracking_id = order.tracking_id

        if not tracking_id:
            raise ToolValidationError("Tracking id not found for this order.")

        shipment = self._store.get_shipment(tracking_id)
        if not shipment:
            raise ToolValidationError("Shipment not found.")

        return TrackShipmentOutput(
            tracking_id=shipment.tracking_id,
            carrier=shipment.carrier,
            status=shipment.status,
            estimated_delivery=shipment.estimated_delivery,
            last_update=shipment.last_update,
        )


class IssueRefundTool(Tool[IssueRefundInput, IssueRefundOutput]):
    name = "issue_refund"
    input_model = IssueRefundInput
    output_model = IssueRefundOutput

    def __init__(self, store: InMemoryStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    async def run(
        self, ctx: ToolContext, tool_input: IssueRefundInput
    ) -> IssueRefundOutput:
        order = self._store.get_order(tool_input.order_id)
        if not order:
            raise ToolValidationError("Order not found.")

        assert_order_belongs_to_user(order, ctx.user_id)
        assert_refund_within_window(order, self._settings)
        assert_refund_amount_allowed(
            order,
            amount_usd=float(tool_input.amount_usd),
            settings=self._settings,
        )

        # Idempotency: if the client repeats the same refund request, don't double-refund.
        if tool_input.idempotency_key:
            if tool_input.idempotency_key in order.refund_idempotency_keys:
                return IssueRefundOutput(
                    order_id=order.order_id,
                    approved=True,
                    refunded_amount_usd=0.0,
                    refund_id="refund_duplicate",
                    message=(
                        "Duplicate refund request detected; no additional refund issued."
                    ),
                )

        # Example eligibility: only shipped/delivered can be refunded here.
        if order.status not in {"shipped", "delivered"}:
            raise ToolPolicyError(
                "Order is not eligible for refund in its current status."
            )

        refund_id = f"rf_{uuid4().hex[:12]}"
        order.refunded_amount_usd += float(tool_input.amount_usd)
        if tool_input.idempotency_key:
            order.refund_idempotency_keys.add(tool_input.idempotency_key)

        return IssueRefundOutput(
            order_id=order.order_id,
            approved=True,
            refunded_amount_usd=float(tool_input.amount_usd),
            refund_id=refund_id,
            message="Refund approved and initiated.",
        )


class UpdateContactTool(Tool[UpdateContactInput, UpdateContactOutput]):
    name = "update_contact"
    input_model = UpdateContactInput
    output_model = UpdateContactOutput

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def run(
        self, ctx: ToolContext, tool_input: UpdateContactInput
    ) -> UpdateContactOutput:
        user = self._store.get_user(ctx.user_id)
        if not user:
            raise ToolValidationError("User not found.")

        updated = self._store.update_user_contact(
            ctx.user_id,
            new_email=str(tool_input.new_email) if tool_input.new_email else None,
            new_phone_e164=tool_input.new_phone_e164,
        )

        return UpdateContactOutput(
            user_id=updated.user_id,
            updated_email=updated.email,
            updated_phone_e164=updated.phone_e164,
            message="Contact information updated.",
        )


class InitiatePasswordResetTool(
    Tool[InitiatePasswordResetInput, InitiatePasswordResetOutput]
):
    name = "initiate_password_reset"
    input_model = InitiatePasswordResetInput
    output_model = InitiatePasswordResetOutput

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def run(
        self, ctx: ToolContext, tool_input: InitiatePasswordResetInput
    ) -> InitiatePasswordResetOutput:
        user = self._store.get_user(ctx.user_id)
        if not user:
            raise ToolValidationError("User not found.")

        # Basic ownership check: only reset for the logged-in user's email.
        if user.email.lower() != str(tool_input.email).lower():
            raise ToolPolicyError(
                "Password reset can only be initiated for the authenticated user's email."
            )

        # In production you'd enqueue an email job and NEVER return the token.
        _ = datetime.now(UTC)

        return InitiatePasswordResetOutput(
            email=user.email,
            initiated=True,
            message="Password reset email has been sent if the account exists.",
        )