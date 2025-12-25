"""
Tool executor (Phase 2).

This is the single "gateway" to tool execution.
Later, the agent will call ONLY this executor.

Responsibilities:
- Allow-list enforcement (tool must be registered)
- Input parsing and validation (Pydantic)
- Timeouts (asyncio.wait_for)
- Retries for transient errors (tenacity)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
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
    """
    A structured representation of a tool invocation request.

    Later, the LLM/agent will propose:
      {"name": "...", "args": {...}}
    and we will convert it into ToolCall.
    """
    name: ToolName
    args: dict[str, Any]


class ToolExecutor:
    def __init__(self, *, registry: ToolRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings

    async def execute(self, ctx: ToolContext, call: ToolCall) -> BaseModel:
        tool = self._registry.get(call.name)
        if not tool:
            raise ToolNotFoundError(f"Tool not registered: {call.name}")

        try:
            parsed_input = tool.input_model.model_validate(call.args)
        except ValidationError as e:
            raise ToolValidationError(str(e)) from e

        return await self._execute_with_controls(tool, ctx, parsed_input)

    @retry(
        retry=retry_if_exception_type(ToolTransientError),
        stop=stop_after_attempt(3),  # overridden below to match settings
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2.0),
        reraise=True,
    )
    async def _execute_with_controls(
        self,
        tool: Any,
        ctx: ToolContext,
        parsed_input: BaseModel,
    ) -> BaseModel:
        """
        Execute with timeout and retry controls.

        Tenacity decorator retries only ToolTransientError.
        """
        # Tenacity stop_after_attempt is static; enforce settings here too.
        # If settings.tool_max_retries = 2, attempts = 1 initial + 2 retries = 3.
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