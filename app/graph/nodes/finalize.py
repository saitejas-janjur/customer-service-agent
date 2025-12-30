"""
Finalize Node.

Prepares the final response state.
"""

from __future__ import annotations

from app.config import Settings
from app.graph.state import AgentState


def finalize_node(state: AgentState, settings: Settings) -> dict:
    last_msg = state["messages"][-1]
    return {"final_answer": last_msg.content}