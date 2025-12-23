"""
Centralized logging setup.

Big-tech style rule: don't scatter logging config across scripts.
This function standardizes format and log level across the repo.
"""

import logging
from typing import Final


_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def setup_logging(level: str) -> None:
    """
    Configure root logging.

    Args:
        level: e.g. "DEBUG", "INFO", "WARNING"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
    )

    # Reduce noise from very chatty libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)