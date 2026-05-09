# api/schemas.py

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    file_name: str
    file_type: str
    total_chunks: int
    success: bool


# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    verdict: str | None = None
    was_revised: bool = False
    reformulated_query: str | None = None
    chunk_count: int = 0
    agent_logs: list[dict] = Field(default_factory=list)
    success: bool = True
    error: str | None = None


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_ready: bool
    vector_count: int = 0
    source_count: int = 0
    llm_provider: str
    environment: str
