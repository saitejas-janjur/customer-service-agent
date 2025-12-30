"""
Persistence Layer.

Responsibilities:
- Provide a checkpointer for LangGraph to save/load state.
- Uses langgraph-checkpoint-sqlite.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import Settings


@contextmanager
def get_checkpointer(settings: Settings) -> Generator[BaseCheckpointSaver, None, None]:
    """
    Context manager to yield a configured checkpointer.
    
    We use a context manager because SqliteSaver needs to manage
    a connection that should be closed cleanly.
    """
    # Ensure directory exists
    settings.state_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(
        str(settings.state_db_path),
        check_same_thread=False
    )
    
    # Construct the saver
    checkpointer = SqliteSaver(conn)
    
    try:
        # Perform setup (create tables) if needed
        # Calling setup() is idempotent in newer LangGraph versions
        checkpointer.setup()
        yield checkpointer
    finally:
        conn.close()