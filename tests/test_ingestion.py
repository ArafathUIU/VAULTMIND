"""Tests for document ingestion."""

from pathlib import Path

from core.config import EmbeddingProvider
from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from ingestion.loader import load_document
from ingestion.vector_store import InMemoryVectorStore


def test_ingestion_modules_import() -> None:
    """Verify ingestion scaffold modules are importable."""
    import ingestion.chunker
    import ingestion.embedder
    import ingestion.loader
    import ingestion.vector_store

    assert ingestion.loader is not None


def test_load_txt_document(tmp_path: Path) -> None:
    file_path = tmp_path / "policy.txt"
    file_path.write_text("Annual leave policy allows paid vacation.", encoding="utf-8")

    document = load_document(file_path)

    assert document.text == "Annual leave policy allows paid vacation."
    assert document.source == "policy.txt"
    assert document.file_type == "txt"


def test_chunk_document_uses_overlap(tmp_path: Path) -> None:
    file_path = tmp_path / "handbook.txt"
    file_path.write_text("one two three four five six seven", encoding="utf-8")
    document = load_document(file_path)

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]
    assert chunks[0].metadata["word_start"] == 0
    assert chunks[1].metadata["word_start"] == 2


def test_local_embedder_returns_normalized_vectors() -> None:
    embedder = Embedder(provider=EmbeddingProvider.LOCAL, dimensions=16)

    embedding = embedder.embed_text("leave policy")

    magnitude = sum(value * value for value in embedding) ** 0.5

    assert len(embedding) == 16
    assert round(magnitude, 6) == 1.0


def test_vector_store_returns_most_relevant_chunk(tmp_path: Path) -> None:
    file_path = tmp_path / "handbook.txt"
    file_path.write_text(
        "annual leave vacation entitlement\n"
        "server database deployment monitoring",
        encoding="utf-8",
    )
    document = load_document(file_path)
    chunks = chunk_document(document, chunk_size=4, chunk_overlap=0)
    embedder = Embedder(provider=EmbeddingProvider.LOCAL, dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)

    store.add(embedder.embed_chunks(chunks))
    results = store.search("vacation leave", top_k=1)

    assert len(store) == 2
    assert results[0].chunk.text == "annual leave vacation entitlement"
