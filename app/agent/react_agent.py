"""
ReAct agent implementation.

This module constructs a LangChain ReAct agent that can:
- call allow-listed business tools (Phase 2)
- call a KB search tool (Phase 1)
- produce a final user answer

We intentionally wrap LangChain in our own class so:
- the rest of the app depends on OUR interface, not LangChain internals
- future migration (LangGraph) is straightforward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent

from app.agent.prompts import build_react_prompt
from app.agent.tools_adapter import build_langchain_tools
from app.config import Settings
from app.llm.models import build_reasoning_llm
from app.tools.audit import ToolAuditLogger
from app.tools.executor import ToolExecutor
from app.tools.factory import build_tool_registry
from app.tools.types import ToolContext


@dataclass(frozen=True)
class AgentRunResult:
    """
    Output of a single agent run.

    - answer: user-facing text
    - intermediate_steps: raw steps for debugging (not for user)
    """
    answer: str
    intermediate_steps: list[Any]


class CustomerServiceReActAgent:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

        # Build registry once; in production this would use DB clients, etc.
        self._registry, _store = build_tool_registry(settings)

        # LLM for reasoning/tool decisions.
        self._llm = build_reasoning_llm(settings)

        # Audit logger for tool calls
        self._audit = ToolAuditLogger(audit_dir=settings.audit_dir)

    async def run(self, *, ctx: ToolContext, user_message: str) -> AgentRunResult:
        """
        Run the agent once.

        Args:
            ctx: authenticated tool context (user_id, request_id)
            user_message: user's message

        Returns:
            AgentRunResult with final answer and intermediate debug steps.
        """
        executor = ToolExecutor(
            registry=self._registry,
            settings=self._s,
            audit_logger=self._audit,
        )

        tools = build_langchain_tools(executor=executor, ctx=ctx, enable_kb_tool=True)
        prompt = build_react_prompt()

        agent = create_react_agent(llm=self._llm, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=int(self._s.agent_max_iterations),
            verbose=bool(self._s.agent_verbose),
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        result: dict[str, Any] = await agent_executor.ainvoke({"input": user_message})

        answer = str(result.get("output", "")).strip()
        steps = result.get("intermediate_steps", []) or []

        return AgentRunResult(answer=answer, intermediate_steps=steps)