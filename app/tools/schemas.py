"""
Tool input/output schemas.

Rules:
- Inputs are validated server-side using Pydantic.
- Outputs are structured and stable for downstream agent logic and UI rendering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import EmailStr

from pydantic import BaseModel, Field, EmailStr, StringConstraints
from typing_extensions import Annotated


OrderId = Annotated[str, StringConstraints(pattern=r"^ord_[a-zA-Z0-9]{6,32}$")]
TrackingId = Annotated[str, StringConstraints(pattern=r"^trk_[a-zA-Z0-9]{6,32}$")]


class GetOrderStatusInput(BaseModel):
    order_id: OrderId


class OrderLineItem(BaseModel):
    sku: str
    name: str
    quantity: int = Field(ge=1)
    unit_price_usd: float = Field(ge=0.0)


class GetOrderStatusOutput(BaseModel):
    order_id: str
    status: Literal["processing", "shipped", "delivered", "cancelled"]
    created_at: datetime
    total_amount_usd: float
    currency: Literal["USD"] = "USD"
    items: list[OrderLineItem]


class TrackShipmentInput(BaseModel):
    # Support either tracking_id or order_id.
    tracking_id: TrackingId | None = None
    order_id: OrderId | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.tracking_id and not self.order_id:
            raise ValueError("Provide either tracking_id or order_id.")


class TrackShipmentOutput(BaseModel):
    tracking_id: str
    carrier: str
    status: Literal["label_created", "in_transit", "out_for_delivery", "delivered"]
    estimated_delivery: datetime | None = None
    last_update: datetime


class IssueRefundInput(BaseModel):
    order_id: OrderId
    amount_usd: float = Field(gt=0.0)
    reason: str = Field(min_length=3, max_length=240)
    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Optional idempotency key to avoid duplicate refunds if the "
            "same request is repeated."
        ),
    )


class IssueRefundOutput(BaseModel):
    order_id: str
    approved: bool
    refunded_amount_usd: float
    currency: Literal["USD"] = "USD"
    refund_id: str
    message: str


class UpdateContactInput(BaseModel):
    new_email: EmailStr | None = None
    new_phone_e164: str | None = Field(
        default=None,
        description="Phone in E.164 format, e.g. +14155552671",
        pattern=r"^\+[1-9]\d{6,14}$",
    )

    def model_post_init(self, __context: object) -> None:
        if not self.new_email and not self.new_phone_e164:
            raise ValueError("Provide at least one of new_email or new_phone_e164.")


class UpdateContactOutput(BaseModel):
    user_id: str
    updated_email: str | None
    updated_phone_e164: str | None
    message: str


class InitiatePasswordResetInput(BaseModel):
    email: EmailStr


class InitiatePasswordResetOutput(BaseModel):
    email: str
    initiated: bool
    message: str