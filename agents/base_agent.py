# agents/base_agent.py

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# AGENT RESULT
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Standardised output from every agent.
    The orchestrator always receives this — never raw strings.
    """
    agent_name: str
    success: bool
    output: Any                          # the actual result
    error: str | None = None             # set if success=False
    latency_ms: float = 0.0              # how long the agent took
    metadata: dict = field(default_factory=dict)  # agent-specific extras

    @property
    def failed(self) -> bool:
        return not self.success


# ─────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all VaultMind agents.

    Every agent must implement run().
    Everything else — LLM calls, timing, error handling,
    logging — is handled here so agents stay focused
    on their one job.

    Usage:
        class RouterAgent(BaseAgent):
            def run(self, **kwargs) -> AgentResult:
                ...
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel | None = None,
    ):
        self.name = name
        self.llm = llm or get_llm()
        self.logger = logging.getLogger(f"vaultmind.agents.{name}")

    # ─────────────────────────────────────────
    # INTERFACE — subclasses must implement
    # ─────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """
        Execute the agent's task.
        Must return an AgentResult — never raise directly.
        """
        ...

    # ─────────────────────────────────────────
    # HELPERS — available to all agents
    # ─────────────────────────────────────────

    def call_llm(self, messages: list[BaseMessage]) -> str:
        """
        Call the LLM and return the response text.
        Handles timing and error logging automatically.
        """
        start = time.time()
        try:
            response = self.llm.invoke(messages)
            elapsed_ms = round((time.time() - start) * 1000, 2)
            self.logger.debug(
                f"{self.name} LLM call completed in {elapsed_ms}ms"
            )
            return response.content.strip()

        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            self.logger.error(
                f"{self.name} LLM call failed after {elapsed_ms}ms: {e}"
            )
            raise

    def ok(self, output: Any, latency_ms: float = 0.0, **metadata) -> AgentResult:
        """Shorthand for a successful AgentResult."""
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def fail(self, error: str, latency_ms: float = 0.0) -> AgentResult:
        """Shorthand for a failed AgentResult."""
        self.logger.error(f"{self.name} failed: {error}")
        return AgentResult(
            agent_name=self.name,
            success=False,
            output=None,
            error=error,
            latency_ms=latency_ms,
        )

    def timed_run(self, **kwargs) -> AgentResult:
        """
        Wraps run() with automatic timing and top-level error catching.
        The orchestrator calls this instead of run() directly.

        This means if an agent crashes unexpectedly, the whole
        pipeline doesn't crash — it returns a clean AgentResult(success=False).
        """
        start = time.time()
        try:
            result = self.run(**kwargs)
            result.latency_ms = round((time.time() - start) * 1000, 2)
            self.logger.info(
                f"{self.name} completed in {result.latency_ms}ms "
                f"— {'OK' if result.success else 'FAILED'}"
            )
            return result

        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            self.logger.exception(f"{self.name} raised an unhandled exception: {e}")
            return self.fail(error=str(e), latency_ms=elapsed_ms)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
