"""Reviewer agent — the nitpicky bastard who makes the code better."""

from __future__ import annotations

import json

from the_bois.agents.base import BaseAgent
from the_bois.memory.ledger import MessageType
from the_bois.utils import parse_json_response

SYSTEM_PROMPT = """\
You are the Reviewer. You evaluate code for correctness and completeness.

RULE 1 (HIGHEST PRIORITY) — FATAL CHECKS (reject if ANY true):
  ☐ Function called but never defined or imported → NameError at runtime
  ☐ Function body is just `pass` or contains TODO instead of real code
  ☐ Task requirements described but not actually implemented
  ☐ Functions from previous tasks are missing (dropped code)
  ☐ Code would fail to parse (SyntaxError)

RULE 2 — LOGIC & CORRECTNESS:
  ☐ Off-by-one errors, unclosed resources, race conditions
  ☐ Attribute access on wrong type (e.g., calling .items() on a list)
  ☐ Exception handling that silently swallows errors

RULE 3 — RUNTIME VERIFICATION:
  ☐ Every attribute/method access uses a name that actually exists on that type
  ☐ Widget/component initialization matches the framework's lifecycle \
(e.g., attributes set in compose() may not exist in __init__())
  ☐ String literals contain only ASCII characters unless Unicode is intentional
  ☐ Test setup creates all objects/attributes the assertions will check

RULE 4 — DATA LIFECYCLE:
  ☐ Every class field is both serialized (save) AND deserialized (load)
  ☐ Round-trip: save → load produces equivalent objects
  ☐ No orphan fields that exist in __init__ but are never persisted

RULE 5 — SECONDARY CHECKS (major, not fatal):
  ☐ Hardcoded paths assuming a specific working directory
  ☐ Missing error handling for file I/O or user input
  ☐ Security issues (injection, unsafe deserialization)

REVIEW PROCEDURE:
1. Read the task description. List what MUST be implemented.
2. Walk each file top-to-bottom. For every function CALL, confirm a DEF exists.
3. Check the checklist above in order (Rule 1 → 2 → 3 → 4 → 5).
4. Compare against previously approved code — nothing should be missing.
5. Render your verdict.

APPROVE only when ALL of these are true:
  ✓ Every function called has a matching def or import
  ✓ No pass placeholders or TODO comments
  ✓ All task requirements fully implemented
  ✓ Code would run without errors

FILE REFERENCES:
- Use ONLY file paths that appear in the task or code being reviewed
- NEVER invent or guess filenames

When in doubt, REJECT. A false rejection costs one retry. A false approval \
ships broken code.

EXAMPLE (for format reference only):

Code contains: `items = data.keys()` then `items.sort()`
Problem: dict_keys has no .sort() method.

{"approved": false, "issues": [{"severity": "critical", "file": "store.py", \
"description": "dict_keys object has no sort() method — will raise \
AttributeError at runtime", "suggestion": "Use sorted(data.keys()) instead"}], \
"summary": "One runtime error: calling .sort() on a dict_keys view."}

(End of example.)

Respond with ONLY valid JSON:
{
  "approved": true or false,
  "issues": [
    {
      "severity": "critical" or "major" or "minor",
      "file": "affected file path",
      "description": "What is wrong",
      "suggestion": "How to fix it"
    }
  ],
  "summary": "Overall assessment"
}
Do NOT include any text outside the JSON object.\
"""


class ReviewerAgent(BaseAgent):
    """Reviews code and provides structured feedback."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def execute(self, input_data: dict) -> dict:
        task = input_data["task"]
        code = input_data["code"]
        context = input_data.get("context", {})
        unchanged_files: list[str] = input_data.get("unchanged_files", [])
        last_validation_error: str | None = input_data.get("last_validation_error")

        files_summary = ""
        for f in code.get("files", []):
            files_summary += f"\n--- {f['path']} ---\n{f['content']}\n"

        prompt = ""

        # Warn the reviewer when its last approval was overridden
        if last_validation_error:
            prompt += (
                "\n⚠️ **YOUR LAST APPROVAL WAS OVERRIDDEN — the code you approved "
                "crashed at runtime:**\n"
                f"{last_validation_error}\n"
                "Do NOT approve code with similar issues. Be extra critical.\n\n"
            )

        prompt += (
            f"Review this code for the following task:\n\n"
            f"**Task:** {task['title']}\n"
            f"**Description:** {task['description']}\n\n"
            f"**Code to review (CHANGED files only):**\n{files_summary}\n"
        )

        if unchanged_files:
            prompt += (
                f"**Unchanged files (identical to approved version, omitted for brevity):**\n"
                + "".join(f"  - {p}\n" for p in unchanged_files)
                + "\n"
            )

        # Include previous task code so reviewer can check for dropped functions
        # and data lifecycle consistency across the codebase.
        if context:
            prev_files = ""
            for result in context.values():
                for f in result.get("code", {}).get("files", []):
                    prev_files += f"\n--- {f['path']} ---\n{f['content']}\n"
            if prev_files:
                prompt += (
                    "\n**Previously approved code (check nothing was dropped "
                    "and data flows are consistent):**\n"
                    f"{prev_files}\n"
                )

        if code.get("explanation"):
            prompt += f"\n**Coder's explanation:** {code['explanation']}\n"

        _REVIEWER_SCHEMA = {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string"},
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["severity", "file", "description", "suggestion"],
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["approved", "issues", "summary"],
        }

        raw = await self.think(
            prompt, json_schema=_REVIEWER_SCHEMA,
            task_description=task.get("description", task.get("title", "")),
            task_id=task.get("id"),
        )

        self.post_message(
            to_agent="coder",
            message_type=MessageType.REVIEW,
            content=raw,
            metadata={"task_id": task.get("id", "unknown")},
        )

        parsed = parse_json_response(raw)
        if parsed:
            if "approved" in parsed:
                return parsed

            # Handle variant schemas from sloppy models
            # e.g. {"consistency": 1, "explanation": "..."}
            if "consistency" in parsed:
                return {
                    "approved": parsed["consistency"] == 1,
                    "issues": parsed.get("issues", []),
                    "summary": parsed.get("explanation", parsed.get("summary", "")),
                }

        # Fallback: assume rejection if we can't parse (safer than blind approval)
        return {
            "approved": False,
            "issues": [{"severity": "critical", "file": "unknown",
                        "description": "Reviewer response could not be parsed.",
                        "suggestion": "Please re-review and respond in the required JSON format."}],
            "summary": f"Could not parse review response. Raw: {raw[:300]}",
        }
