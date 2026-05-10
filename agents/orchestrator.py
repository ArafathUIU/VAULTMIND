# agents/orchestrator.py

import logging
from dataclasses import dataclass, field
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from agents.router_agent import RouterAgent, RouteLabel
from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent
from agents.critic_agent import CriticAgent
from ingestion.vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Shared state passed between every node in the LangGraph pipeline.
    Each agent reads what it needs and writes its output back here.
    """
    # input
    query: str
    top_k: int

    # router output
    route: str | None

    # retriever output
    context: str | None
    chunks: list | None
    reformulated_query: str | None

    # reasoning output
    answer: str | None

    # critic output
    final_answer: str | None
    verdict: str | None
    was_revised: bool

    # error tracking
    error: str | None

    # observability
    agent_logs: Annotated[list[dict], operator.add]


# ─────────────────────────────────────────────
# PIPELINE RESULT
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Clean output returned to the API layer."""
    query: str
    final_answer: str
    verdict: str | None = None
    was_revised: bool = False
    reformulated_query: str | None = None
    chunk_count: int = 0
    agent_logs: list[dict] = field(default_factory=list)
    error: str | None = None
    success: bool = True


@dataclass
class AgentStreamEvent:
    """Structured progress event streamed to the browser while agents run."""
    type: str
    agent: str
    message: str
    status: str
    latency_ms: float | None = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

class VaultOrchestrator:
    """
    LangGraph-powered multi-agent pipeline for VaultMind.

    Flow:
        query
          └─→ router
                ├─→ RETRIEVAL     → retriever → reasoning → critic → END
                ├─→ CONVERSATIONAL → conversational_response → END
                └─→ OUT_OF_SCOPE  → out_of_scope_response → END
    """

    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

        # initialise agents
        self.router    = RouterAgent()
        self.retriever = RetrieverAgent(vector_store=vector_store)
        self.reasoning = ReasoningAgent()
        self.critic    = CriticAgent()

        # compile the graph once at startup
        self.graph = self._build_graph()

    # ─────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────

    def run(self, query: str, top_k: int = 8) -> PipelineResult:
        """
        Run the full pipeline for a user query.
        Returns a PipelineResult — always, even on failure.
        """
        logger.info(f"Pipeline starting — query: '{query[:80]}...'")

        initial_state: PipelineState = {
            "query": query,
            "top_k": top_k,
            "route": None,
            "context": None,
            "chunks": None,
            "reformulated_query": None,
            "answer": None,
            "final_answer": None,
            "verdict": None,
            "was_revised": False,
            "error": None,
            "agent_logs": [],
        }

        try:
            final_state = self.graph.invoke(initial_state)
            return self._build_result(final_state)

        except Exception as e:
            logger.exception(f"Pipeline crashed: {e}")
            return PipelineResult(
                query=query,
                final_answer="Something went wrong while processing your query. Please try again.",
                error=str(e),
                success=False,
            )

    def run_with_events(self, query: str, top_k: int = 8):
        """
        Run the pipeline and yield progress events before returning the final answer.

        This intentionally mirrors the graph flow so the frontend can show the agents
        collaborating in real time without changing the existing non-streaming API.
        """
        logger.info(f"Streaming pipeline starting — query: '{query[:80]}...'")

        state: PipelineState = {
            "query": query,
            "top_k": top_k,
            "route": None,
            "context": None,
            "chunks": None,
            "reformulated_query": None,
            "answer": None,
            "final_answer": None,
            "verdict": None,
            "was_revised": False,
            "error": None,
            "agent_logs": [],
        }

        try:
            yield self._event("started", "router", "I am deciding which specialist should handle this question.")
            update = self._node_router(state)
            self._merge_state(state, update)
            route = state.get("route") or RouteLabel.OUT_OF_SCOPE
            yield self._event(
                "finished",
                "router",
                self._router_message(route),
                self._latest_latency(update),
                {"route": route},
            )

            next_node = self._route_after_router(state)
            if next_node == "retriever":
                yield self._event("started", "retriever", "I am searching the indexed documents for the strongest evidence.")
                update = self._node_retriever(state)
                self._merge_state(state, update)
                yield self._event(
                    "finished",
                    "retriever",
                    self._retriever_message(state),
                    self._latest_latency(update),
                    {
                        "chunk_count": len(state.get("chunks") or []),
                        "reformulated_query": state.get("reformulated_query"),
                    },
                )

                yield self._event("started", "reasoning", "I am building an answer from the retrieved context.")
                update = self._node_reasoning(state)
                self._merge_state(state, update)
                yield self._event(
                    "finished",
                    "reasoning",
                    "I drafted the answer and passed it to the critic for verification.",
                    self._latest_latency(update),
                )

                yield self._event("started", "critic", "I am checking the answer for faithfulness, relevance, and completeness.")
                update = self._node_critic(state)
                self._merge_state(state, update)
                yield self._event(
                    "finished",
                    "critic",
                    self._critic_message(state),
                    self._latest_latency(update),
                    {
                        "verdict": state.get("verdict"),
                        "was_revised": state.get("was_revised", False),
                    },
                )
            elif next_node == "conversational":
                yield self._event("started", "conversational", "I am answering directly because this does not need document retrieval.")
                update = self._node_conversational(state)
                self._merge_state(state, update)
                yield self._event("finished", "conversational", "I prepared a direct conversational response.")
            else:
                yield self._event("started", "scope", "I am checking whether this question belongs to the uploaded documents.")
                update = self._node_out_of_scope(state)
                self._merge_state(state, update)
                yield self._event("finished", "scope", "This question appears outside the uploaded document scope.")

            result = self._build_result(state)
            yield self._final_event(result)

        except Exception as e:
            logger.exception(f"Streaming pipeline crashed: {e}")
            yield self._final_event(
                PipelineResult(
                    query=query,
                    final_answer="Something went wrong while processing your query. Please try again.",
                    error=str(e),
                    success=False,
                )
            )

    # ─────────────────────────────────────────
    # GRAPH NODES
    # ─────────────────────────────────────────

    def _node_router(self, state: PipelineState) -> dict:
        result = self.router.timed_run(query=state["query"])

        log = {"agent": "router", "latency_ms": result.latency_ms}

        if result.failed:
            return {"route": RouteLabel.OUT_OF_SCOPE, "error": result.error, "agent_logs": [log]}

        return {"route": result.output, "agent_logs": [log]}

    def _node_retriever(self, state: PipelineState) -> dict:
        result = self.retriever.timed_run(query=state["query"], top_k=state.get("top_k", 8))

        log = {
            "agent": "retriever",
            "latency_ms": result.latency_ms,
            "chunk_count": result.metadata.get("chunk_count", 0),
            "reformulated_query": result.metadata.get("reformulated_query"),
        }

        if result.failed:
            return {
                "error": result.error,
                "context": None,
                "agent_logs": [log],
            }

        return {
            "context": result.output,
            "chunks": result.metadata.get("chunks", []),
            "reformulated_query": result.metadata.get("reformulated_query"),
            "agent_logs": [log],
        }

    def _node_reasoning(self, state: PipelineState) -> dict:

        # if retrieval failed, skip reasoning
        if not state.get("context"):
            return {
                "answer": "The uploaded documents do not contain enough information to answer this question.",
                "agent_logs": [{"agent": "reasoning", "skipped": True}],
            }

        result = self.reasoning.timed_run(
            query=state["query"],
            context=state["context"],
        )

        log = {"agent": "reasoning", "latency_ms": result.latency_ms}

        if result.failed:
            return {"answer": None, "error": result.error, "agent_logs": [log]}

        return {"answer": result.output, "agent_logs": [log]}

    def _node_critic(self, state: PipelineState) -> dict:

        # if reasoning failed, skip critic
        if not state.get("answer"):
            return {
                "final_answer": "I was unable to generate an answer. Please try rephrasing your question.",
                "verdict": "SKIPPED",
                "agent_logs": [{"agent": "critic", "skipped": True}],
            }

        result = self.critic.timed_run(
            query=state["query"],
            answer=state["answer"],
            context=state["context"],
        )

        log = {
            "agent": "critic",
            "latency_ms": result.latency_ms,
            "verdict": result.metadata.get("verdict"),
            "was_revised": result.metadata.get("was_revised", False),
        }

        return {
            "final_answer": result.output,
            "verdict": result.metadata.get("verdict"),
            "was_revised": result.metadata.get("was_revised", False),
            "agent_logs": [log],
        }

    def _node_conversational(self, state: PipelineState) -> dict:
        from core.llm_factory import get_fast_llm
        from core.prompts import conversational_prompt

        llm = get_fast_llm()
        messages = conversational_prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)

        return {
            "final_answer": response.content.strip(),
            "verdict": "CONVERSATIONAL",
            "agent_logs": [{"agent": "conversational"}],
        }

    def _node_out_of_scope(self, state: PipelineState) -> dict:
        return {
            "final_answer": (
                "That question seems outside the scope of your uploaded documents. "
                "Please ask something related to the files you've uploaded."
            ),
            "verdict": "OUT_OF_SCOPE",
            "agent_logs": [{"agent": "out_of_scope"}],
        }

    # ─────────────────────────────────────────
    # ROUTING CONDITION
    # ─────────────────────────────────────────

    def _route_after_router(self, state: PipelineState) -> str:
        """
        LangGraph conditional edge.
        Decides which node to go to after the router.
        """
        route = state.get("route", RouteLabel.OUT_OF_SCOPE)

        if route == RouteLabel.RETRIEVAL:
            return "retriever"
        elif route == RouteLabel.CONVERSATIONAL:
            return "conversational"
        else:
            return "out_of_scope"

    # ─────────────────────────────────────────
    # GRAPH BUILDER
    # ─────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(PipelineState)

        # register nodes
        graph.add_node("router",         self._node_router)
        graph.add_node("retriever",      self._node_retriever)
        graph.add_node("reasoning",      self._node_reasoning)
        graph.add_node("critic",         self._node_critic)
        graph.add_node("conversational", self._node_conversational)
        graph.add_node("out_of_scope",   self._node_out_of_scope)

        # entry point
        graph.set_entry_point("router")

        # conditional routing after router
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "retriever":      "retriever",
                "conversational": "conversational",
                "out_of_scope":   "out_of_scope",
            }
        )

        # linear edges for retrieval path
        graph.add_edge("retriever",      "reasoning")
        graph.add_edge("reasoning",      "critic")

        # all paths end here
        graph.add_edge("critic",         END)
        graph.add_edge("conversational", END)
        graph.add_edge("out_of_scope",   END)

        return graph.compile()

    # ─────────────────────────────────────────
    # RESULT BUILDER
    # ─────────────────────────────────────────

    def _build_result(self, state: PipelineState) -> PipelineResult:
        return PipelineResult(
            query=state["query"],
            final_answer=state.get("final_answer") or "No answer generated.",
            verdict=state.get("verdict"),
            was_revised=state.get("was_revised", False),
            reformulated_query=state.get("reformulated_query"),
            chunk_count=len(state.get("chunks") or []),
            agent_logs=state.get("agent_logs", []),
            error=state.get("error"),
            success=state.get("error") is None,
        )

    def _event(
        self,
        status: str,
        agent: str,
        message: str,
        latency_ms: float | None = None,
        metadata: dict | None = None,
    ) -> AgentStreamEvent:
        return AgentStreamEvent(
            type="agent",
            agent=agent,
            message=message,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def _final_event(self, result: PipelineResult) -> AgentStreamEvent:
        return AgentStreamEvent(
            type="final",
            agent="vaultmind",
            message=result.final_answer,
            status="finished",
            metadata={
                "query": result.query,
                "final_answer": result.final_answer,
                "verdict": result.verdict,
                "was_revised": result.was_revised,
                "reformulated_query": result.reformulated_query,
                "chunk_count": result.chunk_count,
                "agent_logs": result.agent_logs,
                "success": result.success,
                "error": result.error,
            },
        )

    def _merge_state(self, state: PipelineState, update: dict) -> None:
        for key, value in update.items():
            if key == "agent_logs":
                state["agent_logs"].extend(value)
            else:
                state[key] = value

    def _latest_latency(self, update: dict) -> float | None:
        logs = update.get("agent_logs") or []
        if not logs:
            return None
        return logs[-1].get("latency_ms")

    def _router_message(self, route: str) -> str:
        if route == RouteLabel.RETRIEVAL:
            return "This needs document evidence, so I am sending it to the retriever."
        if route == RouteLabel.CONVERSATIONAL:
            return "This is conversational, so I can answer without searching documents."
        return "This looks outside the uploaded document scope."

    def _retriever_message(self, state: PipelineState) -> str:
        chunk_count = len(state.get("chunks") or [])
        if chunk_count:
            return f"I found {chunk_count} relevant chunk(s) and prepared the context for reasoning."
        return "I could not find enough matching document context for this question."

    def _critic_message(self, state: PipelineState) -> str:
        verdict = state.get("verdict") or "REVIEWED"
        if verdict == "PARSE_ERROR":
            return "The critic could not parse its quality report, so I kept the drafted answer unchanged."
        if state.get("was_revised"):
            return f"Verdict: {verdict}. I revised the answer before sending it to you."
        return f"Verdict: {verdict}. The answer is ready for delivery."
