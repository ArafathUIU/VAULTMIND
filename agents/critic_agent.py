# agents/critic_agent.py

import json

from agents.base_agent import BaseAgent, AgentResult
from core.llm_factory import get_critic_llm
from core.prompts import critic_prompt
from dataclasses import dataclass

@dataclass
class CriticVerdict:
    faithfulness: bool
    relevance: bool
    completeness: bool
    issues: str | None
    verdict: str           # "PASS" or "FAIL"
    revised_answer: str | None


class CriticAgent(BaseAgent):
    """
    Quality control agent.
    Checks the reasoning agent's answer for:
    - Faithfulness  (no hallucinations)
    - Relevance     (actually answers the question)
    - Completeness  (covers the key points)

    Returns the original answer if PASS,
    or a revised answer if FAIL.
    """

    def __init__(self):
        super().__init__(
            name="critic",
            llm=get_critic_llm(),  # temperature=0.0 — strict checking
        )

    def run(self, query: str, answer: str, context: str) -> AgentResult:

        messages = critic_prompt.format_messages(
            context=context,
            query=query,
            answer=answer,
        )

        raw = self.call_llm(messages)

        try:
            parsed = self._parse_verdict(raw)
        except Exception as e:
            self.logger.warning(f"Failed to parse critic response: {e}. Passing answer through.")
            return self.ok(
                output=answer,
                verdict="PARSE_ERROR",
                original_answer=answer,
            )

        # use revised answer if critic flagged issues
        final_answer = (
            parsed.revised_answer
            if parsed.verdict == "FAIL" and parsed.revised_answer
            else answer
        )

        self.logger.info(
            f"Critic verdict: {parsed.verdict} — "
            f"faithfulness={parsed.faithfulness}, "
            f"relevance={parsed.relevance}, "
            f"completeness={parsed.completeness}"
        )

        return self.ok(
            output=final_answer,
            verdict=parsed.verdict,
            faithfulness=parsed.faithfulness,
            relevance=parsed.relevance,
            completeness=parsed.completeness,
            issues=parsed.issues,
            original_answer=answer,
            was_revised=parsed.verdict == "FAIL",
        )

    def _parse_verdict(self, raw: str) -> CriticVerdict:
        # strip any accidental markdown code fences
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)

        return CriticVerdict(
            faithfulness=data["faithfulness"],
            relevance=data["relevance"],
            completeness=data["completeness"],
            issues=data.get("issues"),
            verdict=data["verdict"],
            revised_answer=data.get("revised_answer"),
        )