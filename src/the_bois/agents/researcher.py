"""Researcher agent — the one who actually reads the docs."""

from __future__ import annotations

from the_bois.agents.base import BaseAgent
from the_bois.memory.ledger import MessageType
from the_bois.tools.search import web_search
from the_bois.utils import parse_json_response

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

Respond with ONLY valid JSON:
{
  "findings": "Concise summary of the library/tool and how to use it",
  "key_points": ["import X from Y", "subclass Z to create app", ...],
  "relevant_code": "Complete code example if available (empty string if none)"
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

        # Search the web first
        search_results = await web_search(query, max_results=max_results)

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
            return parsed

        return {
            "findings": raw,
            "key_points": [],
            "relevant_code": "",
        }
