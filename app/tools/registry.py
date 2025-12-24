"""
Tool registry (allow-list).

This is a security control:
- Only registered tools can be executed.
- Each tool has a stable name and typed input/output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.tools.types import ToolContext, ToolName


TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class Tool(Generic[TIn, TOut]):
    """
    Base interface for tools.

    Tools are implemented as async because:
    - production tool calls often hit DBs/APIs
    - FastAPI / LangGraph are async-friendly
    """

    name: ToolName
    input_model: type[TIn]
    output_model: type[TOut]

    async def run(self, ctx: ToolContext, tool_input: TIn) -> TOut:
        raise NotImplementedError


@dataclass(frozen=True)
class RegisteredTool:
    tool: Tool[Any, Any]


class ToolRegistry:
    """
    Central allow-list registry.

    The executor will look up tools here. No registry entry => no execution.
    """

    def __init__(self) -> None:
        self._tools: dict[ToolName, RegisteredTool] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        self._tools[tool.name] = RegisteredTool(tool=tool)

    def get(self, name: ToolName) -> Tool[Any, Any] | None:
        entry = self._tools.get(name)
        return entry.tool if entry else None

    def names(self) -> list[ToolName]:
        return sorted(self._tools.keys())