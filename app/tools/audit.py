"""
Tool audit logging.

Production goal:
- Every tool call is auditable: who, what, when, inputs, outcome.
- Logs are append-only (JSONL), easy to ship to ELK/Splunk/Datadog later.

Security note:
- We do minimal masking here (email/phone). Phase 7 will add robust PII
  redaction before logging/model calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _mask_email(email: str) -> str:
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        return "***@" + domain
    return local[:2] + "***@" + domain


def _mask_phone(phone: str) -> str:
    # Keep country code and last 2 digits.
    if len(phone) < 6:
        return "***"
    return phone[:2] + "***" + phone[-2:]


def _redact(obj: Any) -> Any:
    """
    Very small redaction helper.
    - For dicts: mask keys that look like PII fields.
    - For lists: redact each element.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            lk = k.lower()
            if lk in {"email", "new_email"} and isinstance(v, str):
                out[k] = _mask_email(v)
            elif lk in {"phone", "new_phone_e164", "phone_e164"} and isinstance(v, str):
                out[k] = _mask_phone(v)
            else:
                out[k] = _redact(v)
        return out

    if isinstance(obj, list):
        return [_redact(x) for x in obj]

    return obj


@dataclass(frozen=True)
class ToolAuditEvent:
    timestamp: str
    request_id: str
    user_id: str
    actor: str
    tool_name: str
    args: dict[str, Any]
    status: str  # "success" | "error"
    error_type: str | None
    error_message: str | None
    duration_ms: int | None


class ToolAuditLogger:
    def __init__(self, *, audit_dir: Path) -> None:
        self._path = audit_dir / "tool_calls.jsonl"
        audit_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: ToolAuditEvent) -> None:
        payload = event.__dict__
        payload["args"] = _redact(payload["args"])

        line = json.dumps(payload, ensure_ascii=False)
        self._path.open("a", encoding="utf-8").write(line + "\n")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()