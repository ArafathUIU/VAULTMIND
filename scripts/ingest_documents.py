"""CLI entry point for bulk document ingestion."""

import argparse

from core.config import settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.vector_store import VaultVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk ingest documents into VaultMind.")
    parser.add_argument("directory", nargs="?", default=settings.raw_data_path)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    return parser


def main() -> None:
    """Run the document ingestion pipeline."""
    args = build_parser().parse_args()
    loader = DocumentLoader(raw_data_path=args.directory)
    documents = loader.load_directory(args.directory)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedder = Embedder()
    vector_store = VaultVectorStore(embedder=embedder)
    vector_store.add(embedder.embed_chunks(chunks))

    stats = vector_store.stats()
    print(
        f"Indexed {stats['vector_count']} chunks from "
        f"{stats['source_count']} source document(s)."
    )


if __name__ == "__main__":
    main()
