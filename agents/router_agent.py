# agents/router_agent.py

from agents.base_agent import BaseAgent, AgentResult
from core.llm_factory import get_fast_llm
from core.prompts import router_prompt


# ─────────────────────────────────────────────
# ROUTE LABELS
# ─────────────────────────────────────────────

class RouteLabel:
    RETRIEVAL       = "RETRIEVAL"
    CONVERSATIONAL  = "CONVERSATIONAL"
    OUT_OF_SCOPE    = "OUT_OF_SCOPE"

    ALL = {RETRIEVAL, CONVERSATIONAL, OUT_OF_SCOPE}


# ─────────────────────────────────────────────
# ROUTER AGENT
# ─────────────────────────────────────────────

class RouterAgent(BaseAgent):
    """
    Classifies incoming queries into:
      RETRIEVAL      → needs document search
      CONVERSATIONAL → greeting or small talk
      OUT_OF_SCOPE   → nothing to do with the documents

    Uses a fast/cheap LLM — routing is a simple classification
    task, no need for a powerful model.
    """

    def __init__(self):
        super().__init__(
            name="router",
            llm=get_fast_llm(),
        )

    def run(self, query: str) -> AgentResult:
        if not query or not query.strip():
            return self.fail("Empty query received.")

        messages = router_prompt.format_messages(query=query)
        raw = self.call_llm(messages)

        # normalise — strip whitespace, uppercase
        label = raw.strip().upper()

        # guard against unexpected responses
        if label not in RouteLabel.ALL:
            self.logger.warning(
                f"Unexpected route label '{label}' — defaulting to RETRIEVAL"
            )
            label = RouteLabel.RETRIEVAL

        self.logger.info(f"Query routed → {label}")

        return self.ok(
            output=label,
            query=query,
        )
