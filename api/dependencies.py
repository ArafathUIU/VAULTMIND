# api/dependencies.py

import logging
from functools import lru_cache

from fastapi import HTTPException, status

from ingestion.vector_store import VaultVectorStore
from agents.orchestrator import VaultOrchestrator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# VECTOR STORE — singleton, shared across requests
# ─────────────────────────────────────────────

@lru_cache()
def get_vector_store() -> VaultVectorStore:
    """
    Returns a single VaultVectorStore instance for the app lifetime.
    Currently backed by the in-memory vector store implementation.
    """
    return VaultVectorStore()


# ─────────────────────────────────────────────
# ORCHESTRATOR — singleton, shared across requests
# ─────────────────────────────────────────────

@lru_cache()
def get_orchestrator() -> VaultOrchestrator:
    """
    Returns a single VaultOrchestrator instance.
    Agents are initialised once and reused.
    """
    vector_store = get_vector_store()
    try:
        return VaultOrchestrator(vector_store=vector_store)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
