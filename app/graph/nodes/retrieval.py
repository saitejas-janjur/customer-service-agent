"""
Retrieval Node.

Fetches knowledge base articles if the user asks a policy question.
"""

from __future__ import annotations

from app.config import Settings
from app.graph.state import AgentState
from app.retrieval.kb_loader import load_hybrid_retriever


async def retrieval_node(state: AgentState, settings: Settings) -> dict:
    """
    Retrieve documents based on the last user message.
    """
    query = state["messages"][-1].content
    
    # In production, we might rewrite the query using an LLM here to be standalone.
    # For now, we use the raw user message.
    
    retriever = load_hybrid_retriever(settings)
    hits = retriever.retrieve(query)
    
    # Return updates to the state
    return {"retrieved_docs": hits}