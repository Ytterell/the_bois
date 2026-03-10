"""Researcher agent — the one who actually reads the docs."""

from __future__ import annotations

import logging

from the_bois.agents.base import BaseAgent
from the_bois.contracts import ResearchResult
from the_bois.memory.ledger import MessageType
from the_bois.tools.search import web_search
from the_bois.utils import parse_json_response

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Researcher. Extract actionable dev info from search results.

You may receive content from PyPI, GitHub READMEs, or web search results.
Your job is to distill it into USABLE reference material for a coder.

RULE 1 — EXTRACT REAL API INFO:
  ☐ Identify correct import paths (e.g. "from textual.app import App")
  ☐ Identify key classes, their methods, and constructor parameters
  ☐ Identify required patterns (subclassing, decorators, lifecycle hooks)
  ☐ Include real code examples if present in the results

RULE 2 — BE SKEPTICAL:
  ☐ If results are unrelated to the query topic, say "No relevant results found"
  ☐ Do NOT confuse similarly-named products/services/concepts
  ☐ Only include info you are confident is accurate

RULE 3 — PRIORITIZE CODE OVER PROSE:
  ☐ A working code example is worth more than a paragraph of description
  ☐ If you see code snippets, include them verbatim
  ☐ Focus on: imports, class hierarchy, key methods, event handling

RULE 4 — CONFIDENCE RATING:
  ☐ Rate your confidence in the findings:
    • "high"  — found actual code examples AND correct import paths
    • "medium" — found descriptions/docs but no concrete code examples
    • "low"   — results were unrelated, generic marketing, or you had to
                  guess — say so honestly instead of making things up

RULE 5 — PIP PACKAGE vs IMPORT NAME:
  ☐ Many Python packages have a DIFFERENT pip install name from their import
    name.  Examples: pip install google-api-python-client → import googleapiclient,
    pip install Pillow → import PIL, pip install scikit-learn → import sklearn,
    pip install opencv-python → import cv2, pip install PyYAML → import yaml
  ☐ ALWAYS explicitly state BOTH the pip package name AND the import path
  ☐ Fill in the "pip_packages" field with the correct pip install names
  ☐ If you see a discrepancy or the search results reveal the correct
    mapping, make this VERY prominent in your findings

Respond with ONLY valid JSON:
{
  "findings": "Concise summary of the library/tool and how to use it",
  "key_points": ["import X from Y", "subclass Z to create app", ...],
  "relevant_code": "Complete code example if available (empty string if none)",
  "pip_packages": {"import_name": "pip_package_name"},
  "confidence": "high | medium | low"
}
Do NOT include any text outside the JSON object.\
"""


class ResearcherAgent(BaseAgent):
    """Searches the web and summarizes findings for the team."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def execute(self, input_data: dict) -> dict:
        query = input_data["query"]
        max_results = input_data.get("max_results", 5)
        # Error-driven research gets extra context
        error_context = input_data.get("error_context", "")
        failing_code = input_data.get("failing_code", "")
        task_description = input_data.get("task_description", "")
        known_docs_urls = input_data.get("known_docs_urls") or {}

        # Search the web — pass known docs URLs for targeted crawling
        search_results = await web_search(
            query,
            max_results=max_results,
            known_docs_urls=known_docs_urls,
        )

        # Truncate search results to avoid blowing the context window.
        # With num_ctx=16384 and num_predict=4096, input budget is ~12K tokens
        # (~48K chars).  Leave room for system prompt + memory + ledger +
        # error context.
        max_result_chars = 36_000
        if len(search_results) > max_result_chars:
            search_results = search_results[:max_result_chars] + "\n... (truncated)"

        # Build prompt — richer when triggered by errors
        if error_context:
            prompt = self._build_error_prompt(
                query,
                search_results,
                error_context,
                failing_code,
                task_description,
            )
        else:
            prompt = (
                f"Research query: {query}\n\n"
                f"Web search results:\n{search_results}\n\n"
                f"Summarize the relevant findings for a software development team."
            )

        raw = await self.think(prompt, json_mode=True)

        self.post_message(
            to_agent="all",
            message_type=MessageType.RESEARCH,
            content=raw,
            metadata={"query": query},
        )

        parsed = parse_json_response(raw)
        if parsed and "findings" in parsed:
            # Normalise through the dataclass — handles confidence
            # validation, pip_packages type enforcement, etc.
            return ResearchResult.from_dict(parsed).to_dict()

        return ResearchResult(findings=raw, confidence="low").to_dict()

    @staticmethod
    def _build_error_prompt(
        query: str,
        search_results: str,
        error_context: str,
        failing_code: str,
        task_description: str,
    ) -> str:
        """Build a focused prompt when research is triggered by a validation error.

        Instead of a generic "research this query", we tell the researcher
        exactly what went wrong so it can give a targeted answer.
        """
        parts: list[str] = [
            "A coder is stuck on a validation error and needs your help.\n",
        ]
        if task_description:
            parts.append(f"**Task**: {task_description[:300]}\n")
        if failing_code:
            parts.append(f"**Failing code**:\n```python\n{failing_code[:500]}\n```\n")
        parts.append(f"**Error**:\n{error_context[:600]}\n")
        parts.append(f"**Research query**: {query}\n")
        parts.append(f"\n**Web search results**:\n{search_results}\n")
        parts.append(
            "\nBased on the search results and the error above, provide:\n"
            "1. The EXACT correct import paths and pip package names\n"
            "2. What the coder is doing wrong and how to fix it\n"
            "3. A working code example if available\n\n"
            "If the error says 'import name differs from package name', you MUST "
            "identify the correct pip package name for the import.\n"
            "Respond with ONLY valid JSON in the format specified."
        )
        return "\n".join(parts)
