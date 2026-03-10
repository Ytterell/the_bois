"""Coder agent — the one who actually writes the damn code."""

from __future__ import annotations

import logging
import re

from the_bois.agents.base import BaseAgent
from the_bois.contracts import CodeOutput, FileSpec
from the_bois.memory.ledger import MessageType
from the_bois.tools.validator import sanitize_code
from the_bois.utils import estimate_tokens, parse_delimited_files, strip_markdown_fences

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Coder. You write production-quality, RUNNABLE code.

RULE 1 (HIGHEST PRIORITY) — SELF-CHECK:
Before outputting, verify your code against this checklist:
  ✓ Every function called has a matching `def` or import
  ✓ No `pass` placeholders or TODO comments exist — every function has real code
  ✓ No NameError, ImportError, or SyntaxError would occur at runtime
  ✓ Every attribute access (widget.value, obj.method()) uses a REAL attribute \
that exists on that type — do NOT guess API names
  ✓ The main app file has an entrypoint (see RULE 7)
  ✓ The code can actually be imported AND executed end-to-end
  ✓ Event handlers, callbacks, and message classes use the CORRECT base classes \
and signatures from the framework (see RULE 8)
  ✓ ALL files requested by the task are present in your output
  ✓ Test files import from YOUR generated modules (not phantom packages)
  ✓ Cross-file imports match: if file A imports from file B, file B defines \
what file A expects

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

RULE 6 — REQUEST RESEARCH:
- If you are unsure about a library's API and no API REFERENCE section is
  available (or the reference doesn't cover what you need), output:
  NEEDS_RESEARCH: <specific query about what you need>
  Example: NEEDS_RESEARCH: textual ListView widget events attributes methods
- Do NOT guess API names. If you don't know them, request research.
- Only use this ONCE per attempt. After receiving research results, you MUST
  produce code on the next attempt.

RULE 7 — RUNNABLE ENTRYPOINT:
- The main application file MUST include an entrypoint that can actually run:
  if __name__ == "__main__":
      app = MyApp()
      app.run()  # or equivalent for the framework
- Without this, the code is USELESS. A user must be able to run your code.
- NEVER use blocking calls like input(), time.sleep(), or synchronous I/O \
inside an async/event-driven application. Use the framework's own mechanisms \
for user input (dialogs, input widgets, events).

RULE 8 — API REFERENCE IS LAW:
- When an API REFERENCE section is provided in the prompt, treat it as the
  ABSOLUTE source of truth. Use ONLY the exact class names, method names,
  constructor arguments, and event types shown in the reference.
- If the reference shows `class Foo(Bar)`, you MUST inherit from `Bar`.
- If the reference shows `widget.text` as the attribute, do NOT use `widget.value`.
- If the reference shows events as `SomeWidget.Changed`, handle them as
  `on_some_widget_changed(self, event: SomeWidget.Changed)`.
- Custom messages MUST inherit from the framework's Message/Event base class.
- NEVER mix patterns from different frameworks (e.g. tkinter patterns in a
  textual app, flask patterns in a django app).

WHEN WRITING TESTS:
- Import and instantiate the project's main classes
- Call actual public methods and event handlers, not just helpers
- Use the framework's test harness if available
- Every test must assert something concrete

SANDBOX CONSTRAINTS (tests run in an isolated environment):
- Third-party packages from the project's venv MAY be available
- If your tests fail with "Ran 0 tests", it usually means:
  a) Your test file has an import error that prevents loading
  b) You used pytest fixtures but pytest is not available — use unittest.TestCase
  c) Test functions don't start with test_
- Prefer unittest.TestCase for maximum compatibility
- Use unittest.mock for mocking, not pytest-specific mocking
- Use tempfile.mkdtemp() instead of pytest's tmp_path fixture

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
        scope: str = input_data.get("scope", "")
        plan_tasks: list[dict] = input_data.get("plan_tasks", [])
        workspace_manifest: str = input_data.get("workspace_manifest", "")

        # ── Prompt ordering: least → most important (recency bias) ──
        # Local models attend most to what appears LAST in the prompt.
        # Order: big picture → reference → codebase → history → feedback → TASK
        prompt = ""

        # 0. Big picture — what the overall project is and where this task fits
        if scope or plan_tasks:
            prompt += "\n**🎯 PROJECT OVERVIEW (your task is ONE piece of this):**\n"
            if scope:
                prompt += f"Goal: {scope[:500]}\n"
            if plan_tasks:
                task_id = task.get("id", "")
                prompt += "Tasks in this project:\n"
                for pt in plan_tasks:
                    marker = " ◀ YOU ARE HERE" if pt.get("id") == task_id else ""
                    prompt += f"  {pt.get('id', '?')}: {pt.get('title', '?')}{marker}\n"
            prompt += "\n"

        # 0.5. Workspace manifest — compact summary of existing files and symbols
        if workspace_manifest:
            prompt += (
                "\n**Existing workspace files (DO NOT recreate these unless "
                "you need to modify them):**\n"
                f"{workspace_manifest}\n\n"
            )

        # 1. Research bank (reference material — consult as needed)
        #    Filter by relevance to current task to avoid blowing context
        if research_bank:
            task_words = set(
                w.lower()
                for w in re.findall(
                    r"[a-zA-Z_][a-zA-Z0-9_]*",
                    f"{task.get('title', '')} {task.get('description', '')} "
                    f"{' '.join(task.get('dependencies', []))}",
                )
                if len(w) > 2
            )
            scored: list[tuple[int, str, str]] = []
            for query, ref in research_bank.items():
                ref_lower = f"{query} {ref}".lower()
                score = sum(1 for w in task_words if w in ref_lower)
                scored.append((score, query, ref))
            scored.sort(reverse=True)

            # Include top-scoring entries (at least score > 0), cap at 5
            relevant = [(q, r) for s, q, r in scored if s > 0][:5]
            # Always include all if there are ≤ 3 total entries
            if len(research_bank) <= 3:
                relevant = list(research_bank.items())

            if relevant:
                prompt += (
                    "\n**📚 API REFERENCE (from documentation — use these correct "
                    "names, do NOT guess API names):**\n"
                )
                for _query, ref in relevant:
                    prompt += f"{ref}\n\n"
                prompt += "---\n\n"

        # 2. Existing codebase files (context for what already exists)
        #    This is the biggest compressible chunk.  If the prompt is getting
        #    too fat, progressively compress code to fit the context window.
        if context:
            # Collect all files from context
            context_files: list[dict] = []
            for task_id_key, result in context.items():
                for f in result.get("code", {}).get("files", []):
                    context_files.append({"path": f["path"], "content": f["content"]})

            if context_files:
                # Estimate how many tokens we have left for code context.
                # Budget = input_budget - system_prompt - non-code prompt so far
                # Leave headroom for the rest of the prompt (task desc, feedback, etc.)
                ctx_limit = self.config.num_ctx or 4096
                predict_budget = self.config.num_predict or 2048
                input_budget = ctx_limit - predict_budget
                sys_tokens = estimate_tokens(self.system_prompt)
                prompt_so_far_tokens = estimate_tokens(prompt)
                # Reserve ~800 tokens for task desc, feedback, plan, ledger, memory
                reserved = 800
                code_budget = max(
                    input_budget - sys_tokens - prompt_so_far_tokens - reserved,
                    200,  # absolute minimum
                )

                from the_bois.tools.context import compress_code_context

                compressed_files, comp_level = compress_code_context(
                    context_files,
                    max_tokens=code_budget,
                )

                existing_files = ""
                for f in compressed_files:
                    existing_files += (
                        f"\n---FILE: {f['path']}---\n{f['content']}\n---END---\n"
                    )

                header = (
                    "\n**Current codebase (you MUST preserve ALL existing "
                    "code and add your new functionality):**\n"
                )
                if comp_level >= 2:
                    header = (
                        "\n**Current codebase (COMPRESSED — signatures/names only "
                        "due to context limits; preserve all existing code):**\n"
                    )
                prompt += f"{header}{existing_files}\n"

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
                f"List:\n"
                f"1. Files to create and the main class/function in each\n"
                f"2. EXACT imports needed (use names from the API REFERENCE if provided)\n"
                f"3. Which file has the entrypoint (if __name__ == '__main__')\n"
                f"4. Event handlers / message classes and their EXACT base classes\n"
                f"5. Any edge cases\n"
                f"Be concise — 5-10 lines max. Reference ONLY real API names."
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
            return CodeOutput(
                explanation="Task already done.",
                already_done=True,
            ).to_dict()

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
                explanation = expl_match[len("EXPLANATION:") :].strip()
            return CodeOutput(
                files=[FileSpec.from_dict(f) for f in files],
                explanation=explanation,
                raw_output=raw,
            ).to_dict()

        # Fallback: maybe the model still output JSON (old habit)
        from the_bois.utils import parse_json_response

        parsed = parse_json_response(raw)
        if parsed and "files" in parsed:
            # Strip markdown fences from any file contents
            for f in parsed["files"]:
                f["content"] = sanitize_code(
                    strip_markdown_fences(f.get("content", ""))
                )
            return CodeOutput(
                files=[FileSpec.from_dict(f) for f in parsed["files"]],
                raw_output=raw,
            ).to_dict()

        # Last resort: treat the whole response as a single file
        content = sanitize_code(strip_markdown_fences(raw))
        return CodeOutput(
            files=[
                FileSpec(
                    path=f"{task.get('id', 'output')}.py",
                    content=content,
                )
            ],
            explanation="Raw output — model did not return structured format.",
            raw_output=raw,
        ).to_dict()
