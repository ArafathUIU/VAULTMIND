"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import get_vector_store
from api.schemas import HealthResponse
from core.config import settings
from ingestion.vector_store import VaultVectorStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(vector_store: VaultVectorStore = Depends(get_vector_store)) -> HealthResponse:
    """Return service health and key runtime state."""
    stats = vector_store.stats()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store_ready=vector_store.is_ready,
        vector_count=stats["vector_count"],
        source_count=stats["source_count"],
        llm_provider=settings.llm_provider.value,
        environment=settings.environment,
    )


@router.get("/ready", response_model=HealthResponse)
def ready(vector_store: VaultVectorStore = Depends(get_vector_store)) -> HealthResponse:
    """Return readiness state for document-backed querying."""
    stats = vector_store.stats()
    return HealthResponse(
        status="ready" if vector_store.is_ready else "waiting_for_documents",
        version=settings.app_version,
        vector_store_ready=vector_store.is_ready,
        vector_count=stats["vector_count"],
        source_count=stats["source_count"],
        llm_provider=settings.llm_provider.value,
        environment=settings.environment,
    )
