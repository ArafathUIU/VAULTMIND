# agents/reasoning_agent.py

from agents.base_agent import BaseAgent, AgentResult
from core.llm_factory import get_llm
from core.prompts import reasoning_prompt


class ReasoningAgent(BaseAgent):
    """
    Generates the final answer using retrieved context.
    Never uses prior knowledge — only what's in the context.
    """

    def __init__(self):
        super().__init__(
            name="reasoning",
            llm=get_llm(),  # full model — this is the heavy lifting
        )

    def run(self, query: str, context: str) -> AgentResult:

        if not context or not context.strip():
            return self.fail("No context provided to reasoning agent.")

        messages = reasoning_prompt.format_messages(
            context=context,
            query=query,
        )

        answer = self.call_llm(messages)

        return self.ok(
            output=answer,
            query=query,
            context_length=len(context),
        )
