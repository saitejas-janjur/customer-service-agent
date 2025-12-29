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
    setup_logging(s.log_level)

    # Optional CLI verbosity
    if args.verbose:
        s.agent_verbose = True

    agent = CustomerServiceReActAgent(s)

    ctx = ToolContext(
        user_id=UserId(args.user_id),
        request_id=RequestId(f"req_{uuid4().hex[:12]}"),
        actor="customer",
    )

    result = await agent.run(ctx=ctx, user_message=args.message)

    print("\nANSWER\n------")
    print(result.answer)

    print("\nCONFIDENCE\n----------")
    print(f"confidence={result.confidence:.2f} needs_human={result.needs_human}")
    if result.reasons:
        print("reasons:")
        for r in result.reasons[:6]:
            print("-", r)

    print("\nDEBUG\n-----")
    print("intermediate steps count =", len(result.intermediate_steps))


if __name__ == "__main__":
    asyncio.run(main())