"""Document query endpoint."""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from agents.orchestrator import AgentStreamEvent, VaultOrchestrator
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


@router.post("/stream")
def stream_query_documents(
    request: QueryRequest,
    orchestrator: VaultOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    """Stream multi-agent progress events and the final answer as NDJSON."""

    def event_stream():
        try:
            for event in orchestrator.run_with_events(request.query):
                yield json.dumps(_event_payload(event)) + "\n"
        except ValueError as exc:
            payload = {
                "type": "final",
                "agent": "vaultmind",
                "status": "error",
                "message": str(exc),
                "metadata": {"success": False, "error": str(exc)},
            }
            yield json.dumps(payload) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _event_payload(event: AgentStreamEvent) -> dict:
    return {
        "type": event.type,
        "agent": event.agent,
        "message": event.message,
        "status": event.status,
        "latency_ms": event.latency_ms,
        "metadata": event.metadata,
    }
