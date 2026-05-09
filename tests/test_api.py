"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.dependencies import get_orchestrator, get_vector_store
from api.main import app
from api.schemas import QueryRequest, QueryResponse
from ingestion.vector_store import VaultVectorStore


class FakePipelineResult:
    query = "What is the leave policy?"
    final_answer = "Employees receive annual leave."
    verdict = "PASS"
    was_revised = False
    reformulated_query = "leave policy"
    chunk_count = 1
    agent_logs = [{"agent": "fake"}]
    success = True
    error = None


class FakeOrchestrator:
    def run(self, query: str) -> FakePipelineResult:
        result = FakePipelineResult()
        result.query = query
        return result


def test_api_modules_import() -> None:
    """Verify API scaffold modules are importable."""
    import api.dependencies
    import api.main
    import api.routes.health
    import api.routes.query
    import api.routes.upload
    import api.schemas

    assert api.main is not None


def test_root_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "VaultMind" in response.text


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "vector_store_ready" in response.json()


def test_upload_endpoint_indexes_text_document() -> None:
    vector_store = VaultVectorStore()

    app.dependency_overrides[get_vector_store] = lambda: vector_store
    client = TestClient(app)

    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("policy.txt", b"annual leave vacation policy", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["file_name"] == "policy.txt"
    assert response.json()["total_chunks"] == 1
    assert vector_store.is_ready is True


def test_query_endpoint_returns_orchestrator_result() -> None:
    app.dependency_overrides[get_orchestrator] = lambda: FakeOrchestrator()
    client = TestClient(app)

    try:
        response = client.post("/query", json={"query": "What is the leave policy?"})
    finally:
        app.dependency_overrides.clear()

    payload = response.json()

    assert response.status_code == 200
    assert payload["final_answer"] == "Employees receive annual leave."
    assert payload["verdict"] == "PASS"
    assert payload["agent_logs"] == [{"agent": "fake"}]


def test_query_request_rejects_empty_query() -> None:
    try:
        QueryRequest(query="")
    except ValidationError as exc:
        assert "String should have at least 1 character" in str(exc)
    else:
        raise AssertionError("Expected validation error for empty query")


def test_query_response_defaults_agent_logs() -> None:
    response = QueryResponse(query="q", final_answer="a")

    assert response.agent_logs == []
    assert response.success is True
