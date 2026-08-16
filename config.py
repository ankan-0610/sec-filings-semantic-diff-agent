from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM provider / model
    # Note: this project uses raw HTTP calls via httpx; no provider SDKs required.
    OPENAI_API_KEY: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    MISTRAL_API_KEY: str | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    LLM_MODEL: str = Field(default="mistral-7b-instruct", description="LLM model id used for reasoning.")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # Thresholds
    DRIFT_THRESHOLD: float = Field(default=0.92, ge=-1.0, le=1.0)
    TRIVIAL_THRESHOLD: float = Field(default=0.98, ge=-1.0, le=1.0)

    # EDGAR API
    EDGAR_USER_AGENT: str = Field(default="SEC Filing Semantic Agent contact@yourmail.com")
    EDGAR_RATE_LIMIT_SLEEP: float = Field(default=0.12, ge=0.0)

    # Database
    DB_PATH: str = Field(default="./sec_agent.db")

    # Scheduler
    SCHEDULER_CRON_HOUR: int = Field(default=2, ge=0, le=23)
    SCHEDULER_CRON_MINUTE: int = Field(default=0, ge=0, le=59)
    SCHEDULER_TIMEZONE: str = Field(default="UTC")

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)

    # Optional: how many new section flags trigger reasoning
    MAX_SECTIONS_PER_RUN: int = Field(default=25, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
