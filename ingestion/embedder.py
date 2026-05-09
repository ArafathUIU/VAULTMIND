"""Embedding provider integration for document chunks and queries."""

import hashlib
import logging
import math

from langchain_openai import OpenAIEmbeddings

from core.config import EmbeddingProvider, settings
from ingestion.models import DocumentChunk, EmbeddedChunk

logger = logging.getLogger(__name__)


class Embedder:
    """Create embeddings for text and VaultMind document chunks."""

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.provider = provider or settings.embedding_provider
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self._client: OpenAIEmbeddings | None = None

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        if self.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(text)
        if self.provider == EmbeddingProvider.LOCAL:
            return _embed_local(text, self.dimensions)

        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """Embed chunks while preserving their source metadata."""
        if not chunks:
            logger.warning("embed_chunks called with an empty chunk list")
            return []

        return [EmbeddedChunk(chunk=chunk, embedding=self.embed_text(chunk.text)) for chunk in chunks]

    def _embed_openai(self, text: str) -> list[float]:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        if self._client is None:
            self._client = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=self.model,
                dimensions=self.dimensions,
            )

        return self._client.embed_query(text)


def embed_chunks(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
    """Embed chunks using configured settings."""
    return Embedder().embed_chunks(chunks)


def embed_query(query: str) -> list[float]:
    """Embed a search query using configured settings."""
    return Embedder().embed_text(query)


def _embed_local(text: str, dimensions: int) -> list[float]:
    """Create a deterministic hashed bag-of-words embedding for local development."""
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0")

    vector = [0.0] * dimensions

    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]
