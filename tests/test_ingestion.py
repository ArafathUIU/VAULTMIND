"""Tests for document ingestion."""

from pathlib import Path

from core.config import EmbeddingProvider
from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader, is_supported_file, load_document, validate_file_size
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
    assert document.metadata["file_name"] == "policy.txt"


def test_load_markdown_as_text(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nMarkdown is plain text here.", encoding="utf-8")

    document = load_document(file_path)

    assert "Markdown is plain text" in document.text
    assert document.file_type == "txt"


def test_load_directory_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "second.md").write_text("second document", encoding="utf-8")
    (tmp_path / "image.png").write_text("not supported", encoding="utf-8")

    documents = DocumentLoader(raw_data_path=str(tmp_path)).load_directory()

    assert [document.source for document in documents] == ["first.txt", "second.md"]


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_text("not supported", encoding="utf-8")

    try:
        load_document(file_path)
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported file type")


def test_file_size_validation_raises_for_large_file(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("too large", encoding="utf-8")

    try:
        validate_file_size(file_path, max_mb=0)
    except ValueError as exc:
        assert "File too large" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized file")


def test_is_supported_file() -> None:
    assert is_supported_file("document.pdf") is True
    assert is_supported_file("notes.md") is True
    assert is_supported_file("image.png") is False


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
