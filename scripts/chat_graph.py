"""
CLI to chat with the LangGraph agent.

Run:
    python scripts/chat_graph.py --user-id user_123
"""

from __future__ import annotations

import argparse
import asyncio
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.graph.builder import build_graph
from app.tools.types import RequestId, UserId
from app.utils.logging import setup_logging


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="user_123", type=str)
    args = parser.parse_args()

    s = get_settings()
    setup_logging(s.log_level)

    print(f"Initializing Graph for user {args.user_id}...")
    graph = build_graph(s)

    # In-memory "session"
    conversation_history = []
    
    # Initial State
    # We treat every run as a new "turn" but pass in history if we were persisting it manually.
    # In a real app, LangGraph Checkpointers handle persistence automatically.
    
    print("\n--- Customer Service AI (Type 'q' to quit) ---\n")
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
            
        request_id = f"req_{uuid4().hex[:8]}"
        
        # Build initial input state
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": UserId(args.user_id),
            "request_id": RequestId(request_id),
            "intent": "general", # default
            "retrieved_docs": [],
            "confidence_score": 1.0,
            "needs_human_review": False,
        }

        # Stream the graph updates
        # We use .astream to see nodes executing in real time
        final_answer = ""
        
        async for event in graph.astream(inputs, stream_mode="updates"):
            for node_name, state_update in event.items():
                print(f"   [Graph: {node_name} executed]")
                
                # If tool was called, show it
                if node_name == "tools":
                    # The update contains "messages" which are ToolMessages
                    for m in state_update.get("messages", []):
                        print(f"      Tool Output: {str(m.content)[:100]}...")
                
                # If agent replied, capture it
                if node_name == "finalize":
                    final_answer = state_update.get("final_answer", "")

        print(f"Bot: {final_answer}")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())