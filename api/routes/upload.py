"""Document upload endpoint."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_vector_store
from api.schemas import StoreResetResponse, UploadResponse
from core.config import settings
from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from ingestion.loader import is_supported_file, load_document, validate_file_size
from ingestion.vector_store import VaultVectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    vector_store: VaultVectorStore = Depends(get_vector_store),
) -> UploadResponse:
    """Upload, ingest, embed, and index a document."""
    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload PDF, DOCX, TXT, or Markdown files.",
        )

    raw_dir = Path(settings.raw_data_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / Path(file.filename).name

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_file_size(destination)
        document = load_document(destination)
        chunks = chunk_document(
            document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        embedded_chunks = Embedder().embed_chunks(chunks)
        vector_store.add(embedded_chunks)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        file.file.close()

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        file_name=document.source,
        file_type=document.file_type,
        total_chunks=len(chunks),
        success=True,
    )


@router.delete("", response_model=StoreResetResponse)
def clear_documents(vector_store: VaultVectorStore = Depends(get_vector_store)) -> StoreResetResponse:
    """Clear the in-memory document index."""
    vector_store.clear()
    return StoreResetResponse(message="Document index cleared.", success=True)
