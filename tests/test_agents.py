"""Tests for agent behavior."""

from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.router_agent import RouteLabel
from ingestion.models import DocumentChunk, SearchResult


def test_agent_modules_import() -> None:
    """Verify agent scaffold modules are importable."""
    import agents.base_agent
    import agents.critic_agent
    import agents.orchestrator
    import agents.reasoning_agent
    import agents.retriever_agent
    import agents.router_agent

    assert agents.router_agent is not None


def test_route_labels_are_complete() -> None:
    assert RouteLabel.ALL == {
        RouteLabel.RETRIEVAL,
        RouteLabel.CONVERSATIONAL,
        RouteLabel.OUT_OF_SCOPE,
    }


def test_retriever_formats_internal_search_results() -> None:
    chunk = DocumentChunk(
        text="Employees receive annual leave after probation.",
        source="handbook.pdf",
        chunk_index=2,
        file_type="pdf",
        metadata={"page_number": 5},
    )
    result = SearchResult(chunk=chunk, score=0.84159)

    context = RetrieverAgent._format_context(None, [result])

    assert "Source: handbook.pdf" in context
    assert "Page: 5" in context
    assert "Chunk: 2" in context
    assert "Score: 0.842" in context
    assert "Employees receive annual leave" in context


def test_critic_parses_json_verdict() -> None:
    raw = """
    ```json
    {
      "faithfulness": true,
      "relevance": true,
      "completeness": false,
      "issues": "Missing one condition.",
      "verdict": "FAIL",
      "revised_answer": "Corrected answer."
    }
    ```
    """

    verdict = CriticAgent._parse_verdict(None, raw)

    assert verdict.faithfulness is True
    assert verdict.relevance is True
    assert verdict.completeness is False
    assert verdict.verdict == "FAIL"
    assert verdict.revised_answer == "Corrected answer."


def test_critic_extracts_json_from_extra_text() -> None:
    raw = """
    Here is the quality check:
    {
      "faithfulness": true,
      "relevance": true,
      "completeness": true,
      "issues": null,
      "verdict": "pass",
      "revised_answer": null
    }
    Hope this helps.
    """

    verdict = CriticAgent._parse_verdict(None, raw)

    assert verdict.faithfulness is True
    assert verdict.relevance is True
    assert verdict.completeness is True
    assert verdict.verdict == "PASS"
    assert verdict.revised_answer is None
