"""
Tool factory / wiring.

We keep construction in one place so production deployments can swap:
- InMemoryStore -> PostgresStore
- Add auth/permissions
- Add external API clients (shipping, billing, etc.)
"""

from __future__ import annotations

from app.config import Settings
from app.tools.implementations import (
    GetOrderStatusTool,
    InitiatePasswordResetTool,
    IssueRefundTool,
    TrackShipmentTool,
    UpdateContactTool,
)
from app.tools.registry import ToolRegistry
from app.tools.store import InMemoryStore


def build_tool_registry(settings: Settings) -> tuple[ToolRegistry, InMemoryStore]:
    store = InMemoryStore()
    registry = ToolRegistry()

    registry.register(GetOrderStatusTool(store))
    registry.register(TrackShipmentTool(store))
    registry.register(IssueRefundTool(store, settings))
    registry.register(UpdateContactTool(store))
    registry.register(InitiatePasswordResetTool(store))

    return registry, store