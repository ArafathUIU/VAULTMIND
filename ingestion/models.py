"""Shared data models for the ingestion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """Raw text extracted from an uploaded file."""

    text: str
    source: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable text unit derived from a source document."""

    text: str
    source: str
    chunk_index: int
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk with its vector embedding."""

    chunk: DocumentChunk
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    """A vector search result with similarity score."""

    chunk: DocumentChunk
    score: float


def normalize_source(file_path: str | Path) -> str:
    """Return a stable source name for metadata and citations."""
    return Path(file_path).name
