"""
Answer confidence scoring.

We use a separate low-temp model to evaluate:
- does the answer follow from tool outputs / KB snippets?
- are there signs of guessing or missing evidence?

This is NOT perfect, but it is a practical production pattern:
"generation model" + "judge model" with threshold gating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.models import build_reasoning_llm


class _JudgeOutput(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human: bool
    reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    needs_human: bool
    reasons: list[str]


class AnswerConfidenceScorer:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._llm = build_reasoning_llm(settings)

    async def score(
        self,
        *,
        user_message: str,
        answer: str,
        intermediate_steps: list[Any],
    ) -> ConfidenceResult:
        """
        Score confidence using a judge prompt that must return JSON.

        If JSON parsing fails, fall back to a conservative heuristic.
        """
        evidence = _summarize_steps(intermediate_steps)

        prompt = "\n".join(
            [
                "You are evaluating a customer support assistant answer for reliability.",
                "Return ONLY JSON with schema:",
                '{"confidence": 0.0, "needs_human": false, "reasons": ["..."]}',
                "",
                "User message:",
                user_message,
                "",
                "Assistant answer:",
                answer,
                "",
                "Evidence (tool observations / KB snippets):",
                evidence,
                "",
                "Guidelines:",
                "- High confidence ONLY if answer is supported by evidence.",
                "- If tools errored or evidence is missing, lower confidence.",
                "- If the answer contains guessed specifics (dates, amounts, status), lower confidence.",
                "- needs_human=true if uncertain or user requests sensitive action.",
            ]
        )

        msg = await self._llm.ainvoke(prompt)
        text = (msg.content or "").strip()

        try:
            data = json.loads(text)
            parsed = _JudgeOutput.model_validate(data)
            return ConfidenceResult(
                confidence=float(parsed.confidence),
                needs_human=bool(parsed.needs_human),
                reasons=list(parsed.reasons),
            )
        except Exception:
            # Conservative fallback: if we have no evidence, score low.
            conf = 0.4 if evidence.strip() else 0.2
            return ConfidenceResult(
                confidence=conf,
                needs_human=True,
                reasons=["Judge parsing failed; falling back to conservative score."],
            )


def _summarize_steps(steps: list[Any]) -> str:
    """
    Convert LangChain intermediate steps into a readable summary.

    steps are typically: List[Tuple[AgentAction, observation]]
    We do NOT rely on their internal classes—just stringify safely.
    """
    if not steps:
        return "[no intermediate steps]"

    lines: list[str] = []
    for i, item in enumerate(steps, start=1):
        try:
            action, observation = item
            tool = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", {})
            lines.append(f"{i}. tool={tool} input={tool_input}")
            obs_str = str(observation)
            lines.append(f"   observation={obs_str[:800]}")
        except Exception:
            lines.append(f"{i}. {str(item)[:800]}")
    return "\n".join(lines)