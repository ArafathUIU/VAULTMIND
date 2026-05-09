"""Document query endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from agents.orchestrator import VaultOrchestrator
from api.dependencies import get_orchestrator
from api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    orchestrator: VaultOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """Run a user query through the multi-agent document pipeline."""
    try:
        result = orchestrator.run(request.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return QueryResponse(
        query=result.query,
        final_answer=result.final_answer,
        verdict=result.verdict,
        was_revised=result.was_revised,
        reformulated_query=result.reformulated_query,
        chunk_count=result.chunk_count,
        agent_logs=result.agent_logs,
        success=result.success,
        error=result.error,
    )
