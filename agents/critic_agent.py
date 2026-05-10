# agents/critic_agent.py

import json
import re

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
        data = json.loads(CriticAgent._extract_json_object(raw))

        return CriticVerdict(
            faithfulness=bool(data["faithfulness"]),
            relevance=bool(data["relevance"]),
            completeness=bool(data["completeness"]),
            issues=data.get("issues"),
            verdict=str(data["verdict"]).upper(),
            revised_answer=data.get("revised_answer"),
        )

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        clean = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Critic response did not contain a JSON object.")

        return clean[start : end + 1]
