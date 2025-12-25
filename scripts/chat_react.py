"""
CLI runner for the Phase 3 ReAct agent.

Run examples:
  python scripts/chat_react.py --user-id user_123 --message "Where is my order ord_XYZ78901?"
  python scripts/chat_react.py --user-id user_123 --message "What is the refund window?"
"""

from __future__ import annotations

import argparse
import asyncio
from uuid import uuid4

from app.agent.react_agent import CustomerServiceReActAgent
from app.config import get_settings
from app.tools.types import RequestId, ToolContext, UserId
from app.utils.logging import setup_logging


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, type=str)
    parser.add_argument("--message", required=True, type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    if args.verbose:
        s.agent_verbose = True  # simple override for CLI usage

    setup_logging(s.log_level)

    agent = CustomerServiceReActAgent(s)

    ctx = ToolContext(
        user_id=UserId(args.user_id),
        request_id=RequestId(f"req_{uuid4().hex[:12]}"),
        actor="customer",
    )

    result = await agent.run(ctx=ctx, user_message=args.message)

    print("\nANSWER\n------")
    print(result.answer)

    print("\nDEBUG: intermediate steps count =", len(result.intermediate_steps))


if __name__ == "__main__":
    asyncio.run(main())