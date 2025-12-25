"""
In-memory store for Phase 2.

What this provides:
- A fake backend for orders, shipments, and users
- Deterministic seed data for local development and tests

In production, you'd replace this with repository interfaces backed by:
- Postgres (orders/users)
- A shipping provider API
- An internal billing/refund system
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal


OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]
ShipmentStatus = Literal[
    "label_created",
    "in_transit",
    "out_for_delivery",
    "delivered",
]


@dataclass
class OrderItem:
    sku: str
    name: str
    quantity: int
    unit_price_usd: float


@dataclass
class Order:
    order_id: str
    user_id: str
    status: OrderStatus
    created_at: datetime
    total_amount_usd: float
    items: list[OrderItem]
    tracking_id: str | None = None
    refunded_amount_usd: float = 0.0
    refund_idempotency_keys: set[str] = field(default_factory=set)


@dataclass
class Shipment:
    tracking_id: str
    carrier: str
    status: ShipmentStatus
    last_update: datetime
    estimated_delivery: datetime | None = None


@dataclass
class User:
    user_id: str
    email: str
    phone_e164: str | None = None


class InMemoryStore:
    """
    Simple store with seed data.

    This is deliberately tiny but structured: later you can replace this store
    with a DB adapter without changing tool schemas.
    """

    def __init__(self) -> None:
        now = datetime.now(UTC)

        self.users: dict[str, User] = {
            "user_123": User(
                user_id="user_123",
                email="customer@example.com",
                phone_e164="+14155552671",
            )
        }

        self.shipments: dict[str, Shipment] = {
            "trk_ABC12345": Shipment(
                tracking_id="trk_ABC12345",
                carrier="UPS",
                status="in_transit",
                last_update=now - timedelta(hours=4),
                estimated_delivery=now + timedelta(days=2),
            )
        }

        self.orders: dict[str, Order] = {
            "ord_XYZ78901": Order(
                order_id="ord_XYZ78901",
                user_id="user_123",
                status="shipped",
                created_at=now - timedelta(days=10),
                total_amount_usd=120.00,
                items=[
                    OrderItem(
                        sku="sku_001",
                        name="Wireless Mouse",
                        quantity=1,
                        unit_price_usd=40.00,
                    ),
                    OrderItem(
                        sku="sku_002",
                        name="Mechanical Keyboard",
                        quantity=1,
                        unit_price_usd=80.00,
                    ),
                ],
                tracking_id="trk_ABC12345",
            )
        }

    # ---- User operations ----

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def update_user_contact(
        self,
        user_id: str,
        *,
        new_email: str | None,
        new_phone_e164: str | None,
    ) -> User:
        user = self.users[user_id]
        if new_email:
            user.email = new_email
        if new_phone_e164:
            user.phone_e164 = new_phone_e164
        return user

    # ---- Order operations ----

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    # ---- Shipment operations ----

    def get_shipment(self, tracking_id: str) -> Shipment | None:
        return self.shipments.get(tracking_id)