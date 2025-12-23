from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Environment
    env: str = Field(default="local", env="ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

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

    # Retrieval
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    # Agent behavior
    max_iterations: int = 5
    confidence_threshold: float = 0.75

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()