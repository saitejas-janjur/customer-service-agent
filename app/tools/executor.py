from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.tools.audit import ToolAuditEvent, ToolAuditLogger, now_iso
from app.tools.registry import ToolRegistry
from app.tools.types import (
    ToolContext,
    ToolName,
    ToolNotFoundError,
    ToolTransientError,
    ToolValidationError,
)


@dataclass(frozen=True)
class ToolCall:
    name: ToolName
    args: dict[str, Any]


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        settings: Settings,
        audit_logger: ToolAuditLogger | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._audit = audit_logger

    async def execute(self, ctx: ToolContext, call: ToolCall) -> BaseModel:
        start = time.perf_counter()
        error_type: str | None = None
        error_message: str | None = None

        try:
            tool = self._registry.get(call.name)
            if not tool:
                raise ToolNotFoundError(f"Tool not registered: {call.name}")

            try:
                parsed_input = tool.input_model.model_validate(call.args)
            except ValidationError as e:
                raise ToolValidationError(str(e)) from e

            result = await self._execute_with_controls(tool, ctx, parsed_input)

            self._log_event(
                ctx=ctx,
                call=call,
                status="success",
                error_type=None,
                error_message=None,
                duration_ms=_ms_since(start),
            )
            return result

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)

            self._log_event(
                ctx=ctx,
                call=call,
                status="error",
                error_type=error_type,
                error_message=error_message,
                duration_ms=_ms_since(start),
            )
            raise

    @retry(
        retry=retry_if_exception_type(ToolTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2.0),
        reraise=True,
    )
    async def _execute_with_controls(
        self,
        tool: Any,
        ctx: ToolContext,
        parsed_input: BaseModel,
    ) -> BaseModel:
        attempts_allowed = 1 + int(self._settings.tool_max_retries)
        current_attempt = getattr(self._execute_with_controls, "retry", None)
        if current_attempt and current_attempt.statistics:
            attempt_num = current_attempt.statistics.get("attempt_number", 1)
            if attempt_num > attempts_allowed:
                raise ToolTransientError("Max retries exceeded.")

        try:
            return await asyncio.wait_for(
                tool.run(ctx, parsed_input),
                timeout=float(self._settings.tool_timeout_seconds),
            )
        except asyncio.TimeoutError as e:
            raise ToolTransientError("Tool execution timed out.") from e

    def _log_event(
        self,
        *,
        ctx: ToolContext,
        call: ToolCall,
        status: str,
        error_type: str | None,
        error_message: str | None,
        duration_ms: int | None,
    ) -> None:
        if not self._audit:
            return

        evt = ToolAuditEvent(
            timestamp=now_iso(),
            request_id=str(ctx.request_id),
            user_id=str(ctx.user_id),
            actor=str(ctx.actor),
            tool_name=str(call.name),
            args=dict(call.args),
            status=status,
            error_type=error_type,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        self._audit.log(evt)


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)