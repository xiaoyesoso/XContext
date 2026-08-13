from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    flash_llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    pro_llm_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    database_url: str = "sqlite+aiosqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/0"


# Global settings instance.
settings = Settings()
