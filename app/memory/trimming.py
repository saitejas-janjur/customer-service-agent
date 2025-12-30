"""
Context Window Management.

Strategy:
- Keep SystemMessage (instructions).
- Keep last K messages.
- Ensure we do not cut off halfway through a tool call sequence (ToolCall -> ToolMessage).
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from app.config import Settings


def trim_conversation_history(
    messages: Sequence[BaseMessage],
    settings: Settings
) -> list[BaseMessage]:
    """
    Trim message history to fit within context window limits.
    
    We use LangChain's smart `trim_messages` which handles:
    - Preserving SystemMessages
    - Counting tokens (approximate or explicit)
    - Not breaking tool sequences
    """
    
    # We use a simple "last N messages" strategy here for speed.
    # In strict production, you might use token_counter=len(enc.encode(m.content)).
    # "last_max" includes the system message in the count for `trim_messages`.
    
    return trim_messages(
        messages,
        max_tokens=settings.max_history_messages, # abusing param for count-based trim
        token_counter=lambda x: 1, # count messages, not tokens
        strategy="last",
        start_on="human", # try to start cleanly on a human message
        include_system=True,
        allow_partial=False, 
    )