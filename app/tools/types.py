"""
Shared tool types and exceptions.

Why this exists:
- Tools are a security boundary. We want strict typing, strict allow-listing,
  and predictable error handling.
- Later, LangChain/LangGraph will call into these tools via a single executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType


ToolName = Literal[
    "get_order_status",
    "track_shipment",
    "issue_refund",
    "update_contact",
    "initiate_password_reset",
]

UserId = NewType("UserId", str)
RequestId = NewType("RequestId", str)


@dataclass(frozen=True)
class ToolContext:
    """
    Context passed to every tool call.

    In production this would include:
    - authenticated user id
    - request id / trace id
    - ip address / user agent
    - roles/permissions
    """
    user_id: UserId
    request_id: RequestId
    actor: str = "customer"  # e.g. "customer", "agent", "system"


class ToolError(Exception):
    """Base class for tool errors."""


class ToolValidationError(ToolError):
    """Inputs are invalid or fail server-side validation."""


class ToolPolicyError(ToolError):
    """Action violates business policy (refund caps, windows, etc.)."""


class ToolNotFoundError(ToolError):
    """Tool name not on allow-list or not registered."""


class ToolTransientError(ToolError):
    """Temporary failure; safe to retry (timeouts, network hiccups, etc.)."""