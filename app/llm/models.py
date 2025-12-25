"""
Centralized OpenAI model builders.

Why:
- Avoid duplicating model configuration across the codebase.
- Make behavior consistent (temperature, model choice, etc.).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_reasoning_llm(settings: Settings) -> ChatOpenAI:
    """
    Low-temperature model for reliable tool decisions and structured outputs.
    """
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.reasoning_model,
        temperature=0.0,
    )


def build_response_llm(settings: Settings) -> ChatOpenAI:
    """
    Slightly higher temperature model for natural language responses.
    """
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.response_model,
        temperature=0.2,
    )