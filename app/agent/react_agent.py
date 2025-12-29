"""
ReAct agent implementation.

Updates:
- Uses a robust output parser to handle "chatty" models that add text before "Action:".
- This prevents "Could not parse LLM output" errors that cause iteration loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

from app.agent.confidence import AnswerConfidenceScorer, ConfidenceResult
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
    answer: str
    confidence: float
    needs_human: bool
    reasons: list[str]
    intermediate_steps: list[Any]


class CustomerServiceReActAgent:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._registry, _store = build_tool_registry(settings)
        self._llm = build_reasoning_llm(settings)
        self._audit = ToolAuditLogger(audit_dir=settings.audit_dir)
        self._scorer = AnswerConfidenceScorer(settings)

    async def run(self, *, ctx: ToolContext, user_message: str) -> AgentRunResult:
        executor = ToolExecutor(
            registry=self._registry,
            settings=self._s,
            audit_logger=self._audit,
        )

        tools = build_langchain_tools(executor=executor, ctx=ctx, enable_kb_tool=True)
        
        # We manually render tools to fit the robust prompt
        tool_names = ", ".join([t.name for t in tools])
        
        # Build prompt with all variables filled
        prompt = build_react_prompt()
        
        # Bind the LLM to the stop sequence (critical for ReAct)
        llm_with_stop = self._llm.bind(stop=["\nObservation:"])

        # Construct the agent chain manually to ensure correct parsing
        # (create_react_agent can be brittle with newer models)
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: x["intermediate_steps"],
                "tools": lambda x: render_text_description_and_args(tools),
                "tool_names": lambda x: tool_names,
            }
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=int(self._s.agent_max_iterations),
            verbose=bool(self._s.agent_verbose),
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        try:
            result: dict[str, Any] = await agent_executor.ainvoke({"input": user_message})
            answer = str(result.get("output", "")).strip()
            steps = result.get("intermediate_steps", []) or []
        except Exception as e:
            # Fallback if the agent explodes (e.g. max iterations hit without finish)
            answer = "I apologize, but I encountered an error while processing your request."
            steps = []
            if hasattr(e, "intermediate_steps"):
                steps = e.intermediate_steps

        # --- Confidence Scoring ---
        conf: ConfidenceResult = await self._scorer.score(
            user_message=user_message,
            answer=answer,
            intermediate_steps=steps,
        )

        if conf.confidence < float(self._s.confidence_threshold):
            safe = (
                "I’m not fully confident I can answer that correctly with the "
                "information available. If you share more details (for example, "
                "your order id starting with `ord_...`), I can try again, or I can "
                "escalate this to a human support agent."
            )
            return AgentRunResult(
                answer=safe,
                confidence=conf.confidence,
                needs_human=True,
                reasons=conf.reasons,
                intermediate_steps=steps,
            )

        return AgentRunResult(
            answer=answer,
            confidence=conf.confidence,
            needs_human=conf.needs_human,
            reasons=conf.reasons,
            intermediate_steps=steps,
        )