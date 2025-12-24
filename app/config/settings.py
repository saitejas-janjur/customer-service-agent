from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Environment
    env: str = Field(default="local", env="ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Paths
    project_root: Path = Field(default_factory=_default_project_root)
    raw_data_dir: Path = Field(
        default_factory=lambda: _default_project_root() / "data" / "raw"
    )
    processed_data_dir: Path = Field(
        default_factory=lambda: _default_project_root() / "data" / "processed"
    )

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # LangSmith
    langsmith_api_key: str | None = Field(
        default=None, env="LANGSMITH_API_KEY"
    )
    langsmith_project: str | None = Field(
        default="customer-service-agent",
        env="LANGSMITH_PROJECT",
    )

    # Models
    reasoning_model: str = "gpt-4.1-mini"
    response_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval (hybrid)
    vector_top_k: int = 8
    bm25_top_k: int = 8
    final_top_k: int = 5
    hybrid_alpha: float = 0.6  # weight on vector vs bm25

    # Optional reranking (Phase 1 supports LLM rerank)
    enable_rerank: bool = False
    rerank_top_n: int = 8

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()