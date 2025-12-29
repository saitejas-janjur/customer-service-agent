"""
LangGraph State Schema.

This defines the 'memory' of our agent as it moves through the graph.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.retrieval.hybrid import RetrievedChunk
from app.tools.types import RequestId, UserId


class AgentState(TypedDict):
    # Conversation history (Message list with append semantics)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Context (User info & Request info)
    user_id: UserId
    request_id: RequestId
    
    # Internal Reasoning State
    intent: Literal["general", "account_action", "policy_query"]
    retrieved_docs: list[RetrievedChunk]
    
    # Flow Control
    confidence_score: float
    needs_human_review: bool
    
    # Final Output
    final_answer: str | None