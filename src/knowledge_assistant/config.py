
"""Application configuration for the knowledge assistant."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_knowledge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    request_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()