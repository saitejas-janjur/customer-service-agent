"""
Triage Node.

Classifies user input to route to the correct subgraph:
- policy_query: Needs RAG.
- account_action: Needs Tools.
- general: Chat/Greeting.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import Settings
from app.graph.state import AgentState
from app.llm.models import build_reasoning_llm


class TriageOutput(BaseModel):
    intent: Literal["policy_query", "account_action", "general"] = Field(
        ..., description="The classification of the user's intent."
    )


async def triage_node(state: AgentState, settings: Settings) -> dict:
    """
    Analyze the latest user message to determine intent.
    """
    last_msg = state["messages"][-1]
    user_text = last_msg.content

    llm = build_reasoning_llm(settings)
    structured_llm = llm.with_structured_output(TriageOutput)

    prompt = ChatPromptTemplate.from_template(
        """You are a triage expert for a customer service bot.
        
        Classify the user's intent into one of these categories:
        - 'policy_query': Questions about refunds, shipping times, rules, 'how to', or info.
        - 'account_action': Requests to DO something: check order, refund, update email, track package.
        - 'general': Greetings, thank yous, or out-of-scope chatter.

        User Message: {message}
        """
    )

    chain = prompt | structured_llm
    
    try:
        result = await chain.ainvoke({"message": user_text})
        intent = result.intent
    except Exception:
        # Fallback to general if classification fails
        intent = "general"

    return {"intent": intent}