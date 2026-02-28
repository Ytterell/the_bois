"""Coordinator agent — the boss who decides when the bois are done."""

from __future__ import annotations

import json

from the_bois.agents.base import BaseAgent
from the_bois.memory.ledger import MessageType
from the_bois.utils import parse_json_response

REVIEW_PROMPT = """\
You are the Coordinator. You make the final approve/rework/replan decision.

RULE 1 (HIGHEST PRIORITY) — DEFAULT TO REWORK:
  Default to REWORK unless you are CERTAIN the code is correct.
  A false approval ships broken code. A false rework costs one retry.

RULE 2 — CONVERGENCE CHECK:
  ☐ Every function called has a matching def or import (no NameErrors)
  ☐ No placeholder `pass` bodies or TODO comments remain
  ☐ Code covers the FULL original scope — nothing was skipped
  ☐ Files work together (imports between modules are correct)

RULE 3 — BE DECISIVE:
- APPROVE only if ALL checks pass and you are confident it runs correctly.
- If specific tasks have concrete bugs, REWORK those tasks only.
- If the fundamental approach is wrong, REPLAN.
- NEVER loop forever. Good enough to run > perfect but stuck.

EXAMPLE (for format reference only):

Task results: task_1 approved, task_2 has undefined function calls.

{"decision": "rework", "reason": "task_2 calls render_sidebar() which is \
never defined — will crash at runtime", "tasks_to_rework": ["task_2"]}

(End of example.)

Respond with ONLY valid JSON:
{
  "decision": "approve" or "rework" or "replan",
  "reason": "Clear explanation of your decision",
  "tasks_to_rework": ["task_id_1"]
}
Do NOT include any text outside the JSON object.\
"""

SCOPE_ANALYSIS_PROMPT = """\
You are the Coordinator. Analyze the project scope before any code is written.

PROCEDURE:
1. Read the scope. Identify ambiguities, missing details, implicit requirements.
2. Produce a refined scope a developer can implement without guessing.
3. Decide if research is needed.

RULES:
- The refined_scope must be self-contained — someone reading ONLY it should \
fully understand what to build
- DEFAULT TO RESEARCH. If the scope names ANY specific third-party library, \
framework, or API (anything not in the Python standard library), set \
needs_research = true. The coding team WILL hallucinate wrong API names \
unless they have actual documentation. Only skip research for projects \
that use exclusively standard-library modules.
- research_queries should target the specific APIs needed. \
Good: "textual TUI widgets layout compose API", "FastAPI route decorators". \
Bad: "how to code", "python tutorial".
- Generate 1-3 queries focused on the MAIN library's widget/component/API \
reference — the team needs to know correct class names, method signatures, \
and attribute names.

Respond with ONLY valid JSON:
{
  "refined_scope": "Clear, detailed, unambiguous scope with explicit requirements",
  "needs_research": true or false,
  "research_queries": ["specific query 1", "specific query 2"]
}
Do NOT include any text outside the JSON object.\
"""


class CoordinatorAgent(BaseAgent):
    """Makes convergence decisions for the whole system."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_prompt = REVIEW_PROMPT

    @property
    def system_prompt(self) -> str:
        return self._active_prompt

    async def analyze_scope(self, scope: str) -> dict:
        """Analyze a raw scope for ambiguity and decide if research is needed."""
        self._active_prompt = SCOPE_ANALYSIS_PROMPT
        try:
            prompt = (
                f"Analyze this project scope and produce a refined version:\n\n"
                f"{scope}"
            )

            _SCOPE_SCHEMA = {
                "type": "object",
                "properties": {
                    "refined_scope": {"type": "string"},
                    "needs_research": {"type": "boolean"},
                    "research_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["refined_scope", "needs_research", "research_queries"],
            }

            raw = await self.think(prompt, json_schema=_SCOPE_SCHEMA, scope=scope)

            self.post_message(
                to_agent="all",
                message_type=MessageType.SYSTEM,
                content=raw,
                metadata={"phase": "scope_analysis"},
            )

            parsed = parse_json_response(raw)
            if parsed and "refined_scope" in parsed:
                return parsed

            # Fallback: use original scope, skip research
            return {
                "refined_scope": scope,
                "needs_research": False,
                "research_queries": [],
            }
        finally:
            self._active_prompt = REVIEW_PROMPT

    async def execute(self, input_data: dict) -> dict:
        scope = input_data["scope"]
        plan = input_data.get("plan", {})
        results = input_data.get("results", {})
        task_ids = [tid for tid in results if not tid.startswith("_")]

        results_summary = ""
        # Deduplicate files by path — when multiple tasks produce the
        # same file, only keep the latest version.  Without this the
        # coordinator gets N copies of main.py and blows its context.
        code_by_path: dict[str, str] = {}
        for task_id, result in results.items():
            task = result.get("task", {})
            review = result.get("review", {})
            results_summary += (
                f"\n**{task_id}: {task.get('title', 'Unknown')}**\n"
                f"  Review: {'✓ Approved' if review.get('approved') else '✗ Issues found'}\n"
                f"  Summary: {review.get('summary', 'N/A')}\n"
                f"  Iterations used: {result.get('iterations', '?')}\n"
            )
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content:
                    code_by_path[path] = content  # last writer wins

        code_summary = ""
        for path, content in code_by_path.items():
            code_summary += f"\n--- {path} ---\n{content}\n"

        prompt = (
            f"Review the completed work for this project:\n\n"
            f"**Original scope:** {scope}\n\n"
            f"**Task results:**\n{results_summary}\n\n"
        )

        if code_summary:
            prompt += (
                f"**Final code output:**\n{code_summary}\n\n"
                f"Check that:\n"
                f"- ALL functions called are defined (no NameErrors)\n"
                f"- No placeholder/pass bodies or TODOs remain\n"
                f"- The code covers the full original scope\n\n"
            )

        prompt += "Should we approve, rework specific tasks, or replan entirely?"

        _DECISION_SCHEMA = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "rework", "replan"]},
                "reason": {"type": "string"},
                "tasks_to_rework": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["decision", "reason", "tasks_to_rework"],
        }

        raw = await self.think(prompt, json_schema=_DECISION_SCHEMA, scope=scope)

        self.post_message(
            to_agent="all",
            message_type=MessageType.DECISION,
            content=raw,
        )

        parsed = parse_json_response(raw)
        if parsed:
            # Normalize: models sometimes use "response"/"status" instead of "decision"
            if "decision" not in parsed:
                for alt_key in ("response", "status"):
                    if alt_key in parsed:
                        parsed["decision"] = parsed.pop(alt_key)
                        break

            if "decision" in parsed:
                return parsed

            # Handle variant schema: {"message": "..."} without decision key
            if "message" in parsed:
                msg = parsed["message"].lower()
                # Infer decision from message content
                _APPROVE_SIGNALS = (
                    "approve", "complete", "acceptable", "correct",
                    "satisf", "no issue", "no remaining", "meets",
                    "looks good", "well-implemented", "fully implemented",
                    "covers the full", "all functions",
                )
                if any(w in msg for w in _APPROVE_SIGNALS):
                    return {
                        "decision": "approve",
                        "reason": parsed["message"][:200],
                        "tasks_to_rework": [],
                    }
                return {
                    "decision": "rework",
                    "reason": parsed["message"][:200],
                    "tasks_to_rework": parsed.get("tasks_to_rework", []),
                }

        # Default to rework if we can't parse — safer than blind approval
        return {
            "decision": "rework",
            "reason": f"Could not parse coordinator response. Defaulting to rework. Raw: {raw[:200]}",
            "tasks_to_rework": task_ids,
        }
