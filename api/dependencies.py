# api/dependencies.py

import logging
from functools import lru_cache

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
    return VaultOrchestrator(vector_store=vector_store)
