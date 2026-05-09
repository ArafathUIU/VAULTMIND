"""Text splitting and semantic chunking utilities."""

from ingestion.models import Document, DocumentChunk


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """Split a document into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = document.text.split()
    if not words:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        metadata = {
            **document.metadata,
            "word_start": start,
            "word_end": end,
        }

        chunks.append(
            DocumentChunk(
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                file_type=document.file_type,
                metadata=metadata,
            )
        )

        if end == len(words):
            break

        start += step
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """Split multiple documents into chunks."""
    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))

    return chunks
