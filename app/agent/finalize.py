"""
Agent finalization / recovery.

Why this exists:
- Different LangChain versions behave differently when an agent hits
  max_iterations. Some return a generic "Agent stopped..." message.
- In production, you want a best-effort final answer using the evidence
  already gathered (tool outputs / KB snippets) rather than a generic failure.

Security / safety:
- We do not include chain-of-thought. We only use tool observations (JSON).
- If evidence is insufficient, we ask a clarifying question.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.llm.models import build_response_llm


_STOP_PHRASES = (
    "Agent stopped due to iteration limit or time limit",
    "stopped due to iteration limit",
)


def needs_recovery(answer: str) -> bool:
    """
    Decide if we should run the recovery finalizer.
    """
    a = (answer or "").strip()
    if not a:
        return True
    return any(p.lower() in a.lower() for p in _STOP_PHRASES)


async def recover_final_answer(
    *,
    settings: Settings,
    user_message: str,
    intermediate_steps: list[Any],
) -> str:
    """
    Produce a best-effort final answer using gathered evidence.
    """
    llm = build_response_llm(settings)

    evidence = _extract_evidence(intermediate_steps)

    prompt = "\n".join(
        [
            "You are a customer service assistant.",
            "You must answer the user's question using ONLY the evidence below.",
            "",
            "RULES:",
            "- If evidence is insufficient, say what you still need and ask 1-2 clarifying questions.",
            "- Do not invent order status, shipment details, dates, or refund eligibility.",
            "- If evidence includes KB snippets with numbers [1], [2], cite them like [1] in the answer.",
            "",
            "USER QUESTION:",
            user_message,
            "",
            "EVIDENCE (tool outputs and KB snippets):",
            evidence,
            "",
            "Write a concise, helpful final answer:",
        ]
    )

    msg = await llm.ainvoke(prompt)
    return (msg.content or "").strip()


def _extract_evidence(steps: list[Any]) -> str:
    """
    Convert intermediate steps into a compact evidence block.

    We attempt to parse tool observations as JSON to make them readable and stable.
    """
    if not steps:
        return "[no tool evidence collected]"

    lines: list[str] = []
    for i, item in enumerate(steps, start=1):
        try:
            action, observation = item
            tool = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", "")
            lines.append(f"{i}. TOOL: {tool}")
            lines.append(f"   INPUT: {tool_input}")

            obs_text = str(observation)

            # Try to pretty-print JSON observations
            parsed = None
            try:
                parsed = json.loads(obs_text)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) or isinstance(parsed, list):
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)[:2000]
                lines.append("   OUTPUT(JSON):")
                lines.append(pretty)
            else:
                lines.append("   OUTPUT:")
                lines.append(obs_text[:2000])

        except Exception:
            lines.append(f"{i}. {str(item)[:2000]}")

        lines.append("")

    return "\n".join(lines).strip()