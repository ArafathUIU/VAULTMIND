"""Vector store abstraction for FAISS and Pinecone."""

import math

from ingestion.embedder import Embedder
from ingestion.models import EmbeddedChunk, SearchResult


class InMemoryVectorStore:
    """Small vector store useful for local development and tests."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self._items: list[EmbeddedChunk] = []

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Store embedded chunks."""
        self._items.extend(embedded_chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the most similar chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedder.embed_text(query)
        results = [
            SearchResult(chunk=item.chunk, score=cosine_similarity(query_embedding, item.embedding))
            for item in self._items
        ]
        results.sort(key=lambda result: result.score, reverse=True)

        return results[:top_k]

    def __len__(self) -> int:
        return len(self._items)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
