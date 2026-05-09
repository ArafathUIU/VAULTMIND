# core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum
from functools import lru_cache


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class VectorStoreType(str, Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────
    app_name: str = "VaultMind"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = Field(default="development")  # development | staging | production

    # ── LLM ───────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"

    gemini_api_key: str = Field(default="")
    gemini_model: str = "gemini-1.5-pro"
    gemini_fast_model: str = "gemini-1.5-flash"

    anthropic_api_key: str = Field(default="")
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_fast_model: str = "claude-3-5-haiku-20241022"

    groq_api_key: str = Field(default="")
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"

    # ── Embeddings ────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Vector Store ──────────────────────────────────────
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    faiss_index_path: str = "data/vector_index"

    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = "vaultmind"
    pinecone_environment: str = Field(default="")

    # ── Ingestion ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"

    # ── API ───────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50

    # ── Observability ─────────────────────────────────────
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = "vaultmind"
    langsmith_tracing: bool = False

    # ── Evaluation ────────────────────────────────────────
    eval_sample_size: int = 20


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# single import used everywhere:  from core.config import settings
settings = get_settings()
