"""Coder agent — the one who actually writes the damn code."""

from __future__ import annotations

from the_bois.agents.base import BaseAgent
from the_bois.memory.ledger import MessageType
from the_bois.tools.validator import sanitize_code
from the_bois.utils import parse_delimited_files, strip_markdown_fences

SYSTEM_PROMPT = """\
You are the Coder. You write production-quality code.

RULE 1 (HIGHEST PRIORITY) — SELF-CHECK:
Before outputting, verify:
  ✓ Every function called has a matching `def` or import
  ✓ No `pass` placeholders or TODO comments exist
  ✓ No NameError, ImportError, or SyntaxError would occur at runtime
  ✓ Every attribute access (widget.value, obj.method()) uses a REAL attribute \
that exists on that type — do NOT guess API names

RULE 2 — COMPLETE CODE ONLY:
- ALL necessary code — no placeholders, no stubs, no "implement later"
- Every function body must have a real implementation
- If you receive review feedback, address EVERY issue

RULE 3 — PRESERVE EXISTING CODE:
- Include ALL existing functions when modifying a file
- NEVER drop or omit previously implemented functions
- Output the COMPLETE, FINAL version of each file you modify

RULE 4 — ONLY OUTPUT CHANGED FILES:
- Do NOT re-output files you are not modifying
- NEVER output diffs or patches — always complete file content

RULE 5 — ALREADY DONE:
- If the task is ALREADY fully satisfied by existing code: respond ALREADY_DONE
- Only use this if 100% certain. NEVER use after receiving rejection feedback.

WHEN WRITING TESTS:
- Import and instantiate the project's main classes
- Call actual public methods and event handlers, not just helpers
- Use the framework's test harness if available
- Every test must assert something concrete

OUTPUT FORMAT (use exactly this — no JSON, no markdown fences):

EXPLANATION: Brief explanation of your approach

---FILE: relative/path/to/file.ext---
<complete file content>
---END---

Multiple files: repeat the ---FILE: / ---END--- blocks.

EXAMPLE (for format reference only):

Task: "Implement a key-value store with get, set, delete, and list_keys"

EXPLANATION: Simple dict-backed store with type hints and docstrings.

---FILE: store.py---
from __future__ import annotations


class KeyValueStore:
    # Thread-unsafe in-memory key-value store.

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        return sorted(self._data)
---END---

(End of example — your output must follow this exact format.)\
"""


class CoderAgent(BaseAgent):
    """Takes tasks and produces code implementations."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def execute(self, input_data: dict) -> dict:
        task = input_data["task"]
        feedback = input_data.get("feedback")
        context = input_data.get("context", {})

        retry_hint = input_data.get("retry_hint", "")

        failure_history: list[str] = input_data.get("failure_history", [])
        research_bank: dict[str, str] = input_data.get("research_bank", {})

        # ── Prompt ordering: least → most important (recency bias) ──
        # Local models attend most to what appears LAST in the prompt.
        # Order: reference → codebase context → history → feedback → TASK
        prompt = ""

        # 1. Research bank (reference material — consult as needed)
        if research_bank:
            prompt += "\n**📚 API REFERENCE (from documentation — use these correct "\
                "names, do NOT guess API names):**\n"
            for _query, ref in research_bank.items():
                prompt += f"{ref}\n\n"
            prompt += "---\n\n"

        # 2. Existing codebase files (context for what already exists)
        if context:
            existing_files = ""
            for task_id, result in context.items():
                for f in result.get("code", {}).get("files", []):
                    existing_files += (
                        f"\n---FILE: {f['path']}---\n{f['content']}\n---END---\n"
                    )

            if existing_files:
                prompt += (
                    "\n**Current codebase (you MUST preserve ALL existing "
                    "code and add your new functionality):**\n"
                    f"{existing_files}\n"
                )

        # 3. Failure history (brief — what went wrong before)
        if failure_history:
            prompt += "\n⚠️ **PREVIOUS ATTEMPTS (same task — do NOT repeat these mistakes):**\n"
            for entry in failure_history:
                prompt += f"  • {entry}\n"
            prompt += "\n"

        if retry_hint:
            prompt += f"\n⚠️ {retry_hint}\n\n"

        # 4. Feedback / review issues (if retry — what to fix)
        if feedback:
            issues = feedback.get("issues", [])
            if issues:
                prompt += "\n**Review feedback to address:**\n"
                for issue in issues:
                    prompt += f"- [{issue.get('severity', 'unknown')}] {issue.get('description', '')}\n"
                    if issue.get("suggestion"):
                        prompt += f"  Suggestion: {issue['suggestion']}\n"

            # Inject research findings so the coder has correct API info
            research = feedback.get("research_context", "")
            if research:
                prompt += (
                    "\n**❗ API REFERENCE (from docs — use this, your previous "
                    "attempt used wrong API names):**\n"
                    f"{research}\n"
                )

        # 5. TASK DESCRIPTION — LAST (this is what the model should focus on)
        prompt += (
            f"\nImplement this task:\n\n"
            f"**Title:** {task['title']}\n"
            f"**Description:** {task['description']}\n"
        )

        # ── Two-phase: plan first, code second (skip on retries) ──
        # On first attempt the model plans before coding, giving it a
        # scaffold to follow.  On retries (feedback present) we skip
        # straight to coding from the feedback — planning twice wastes
        # time when the model already knows what went wrong.
        if not feedback:
            plan_prompt = (
                f"Before coding, describe your implementation plan for this task:\n"
                f"Title: {task['title']}\n"
                f"Description: {task['description']}\n\n"
                f"List: what files you'll create, what classes/functions each will "
                f"contain, what imports you need, and any edge cases to handle.\n"
                f"Be concise — 5-10 lines max."
            )
            # Use the base think() (non-streaming, cheap) for planning
            plan = await self.think(
                plan_prompt,
                task_description=task.get("description", task.get("title", "")),
                task_id=task.get("id"),
            )
            prompt += (
                f"\n**YOUR IMPLEMENTATION PLAN (follow this exactly):**\n"
                f"{plan.strip()}\n\n"
                f"Now implement the plan above. Output complete files only.\n"
            )

        # Stream coder output for progress visibility
        raw = await self.think_stream(
            prompt,
            task_description=task.get("description", task.get("title", "")),
            task_id=task.get("id"),
        )

        # Check for ALREADY_DONE signal before posting/parsing.
        # The model sometimes writes "EXPLANATION: ... ALREADY_DONE" instead
        # of starting with it, so we search the first 300 chars.
        _head = raw.strip()[:300].upper()
        if "ALREADY_DONE" in _head and "---FILE:" not in _head:
            self.post_message(
                to_agent="reviewer",
                message_type=MessageType.CODE,
                content="ALREADY_DONE",
                metadata={"task_id": task.get("id", "unknown")},
            )
            return {"files": [], "explanation": "Task already done.", "already_done": True}

        self.post_message(
            to_agent="reviewer",
            message_type=MessageType.CODE,
            content=raw,
            metadata={"task_id": task.get("id", "unknown")},
        )

        # Parse the delimiter-based format
        files = parse_delimited_files(raw)
        if files:
            # Sanitise unicode garbage before anything else touches it
            for f in files:
                f["content"] = sanitize_code(f["content"])
            # Extract explanation if present
            explanation = ""
            expl_match = raw.split("---FILE:")[0].strip()
            if expl_match.upper().startswith("EXPLANATION:"):
                explanation = expl_match[len("EXPLANATION:"):].strip()
            return {"files": files, "explanation": explanation}

        # Fallback: maybe the model still output JSON (old habit)
        from the_bois.utils import parse_json_response
        parsed = parse_json_response(raw)
        if parsed and "files" in parsed:
            # Strip markdown fences from any file contents
            for f in parsed["files"]:
                f["content"] = sanitize_code(strip_markdown_fences(f.get("content", "")))
            return parsed

        # Last resort: treat the whole response as a single file
        content = sanitize_code(strip_markdown_fences(raw))
        return {
            "files": [
                {
                    "path": f"{task.get('id', 'output')}.py",
                    "content": content,
                }
            ],
            "explanation": "Raw output — model did not return structured format.",
        }
