"""Retriever agent for query reformulation and vector search."""

from agents.base_agent import BaseAgent, AgentResult
from core.llm_factory import get_fast_llm
from core.prompts import retriever_prompt
from ingestion.models import SearchResult
from ingestion.vector_store import InMemoryVectorStore


class RetrieverAgent(BaseAgent):
    """
    Two responsibilities:
    1. Reformulate the user query into a better search query
    2. Search the vector store and return relevant chunks
    """

    def __init__(self, vector_store: InMemoryVectorStore):
        super().__init__(
            name="retriever",
            llm=get_fast_llm(),
        )
        self.vector_store = vector_store

    def run(self, query: str, top_k: int = 8) -> AgentResult:

        if len(self.vector_store) == 0:
            return self.fail(
                "Vector store is not initialised. Please upload a document first."
            )

        # step 1 — rewrite query for better retrieval
        messages = retriever_prompt.format_messages(query=query)
        reformulated_query = self.call_llm(messages)
        self.logger.info(f"Reformulated query: '{reformulated_query}'")

        # step 2 — search vector store
        results = self.vector_store.search(
            query=reformulated_query,
            top_k=top_k,
        )

        if not results:
            return self.fail("No relevant chunks found in the vector store.")

        # format context for reasoning agent
        context = self._format_context(results)

        return self.ok(
            output=context,
            chunks=[result.chunk for result in results],
            results=results,
            reformulated_query=reformulated_query,
            chunk_count=len(results),
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        """
        Formats retrieved chunks into a clean context string
        that the reasoning agent can read.

        Output format:
            [1] Source: report.pdf | Page: 3 | Chunk: 0 | Score: 0.842
            ...chunk text...
        """
        parts = []
        for index, result in enumerate(results, 1):
            chunk = result.chunk
            page = chunk.metadata.get("page") or chunk.metadata.get("page_number")
            page_label = f" | Page: {page}" if page is not None else ""
            score = round(result.score, 3)

            parts.append(
                f"[{index}] Source: {chunk.source}{page_label} "
                f"| Chunk: {chunk.chunk_index} | Score: {score}\n"
                f"{chunk.text.strip()}"
            )

        return "\n\n".join(parts)
