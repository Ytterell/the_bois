"""Mock Ollama client for dry-run testing.

Returns deterministic canned responses so the full pipeline can be
exercised in seconds without touching a real model.  Detects which
agent is calling by sniffing the system prompt and returns the
appropriate fixture.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from the_bois.models.ollama import OllamaResponse

log = logging.getLogger(__name__)

# ── Canned response fixtures ──────────────────────────────────────── #

_SCOPE_ANALYSIS = json.dumps({
    "refined_scope": (
        "Implement a Python module `adder.py` containing a function "
        "`add(a: int, b: int) -> int` that returns the sum of two integers. "
        "Include a `test_adder.py` with pytest tests covering positive numbers, "
        "negative numbers, and zero."
    ),
    "needs_research": False,
    "research_queries": [],
})

_TASK_PLAN = json.dumps({
    "tasks": [
        {
            "id": "task_1",
            "title": "Implement adder module with tests",
            "description": (
                "Create `adder.py` with function `add(a: int, b: int) -> int` "
                "that returns `a + b`. Create `test_adder.py` with tests: "
                "test_add_positive (assert add(2, 3) == 5), "
                "test_add_negative (assert add(-1, -2) == -3), "
                "test_add_zero (assert add(0, 0) == 0), "
                "test_add_mixed (assert add(-1, 5) == 4)."
            ),
            "dependencies": [],
        }
    ]
})

_CODER_OUTPUT = """\
EXPLANATION: Simple adder function with comprehensive tests.

---FILE: adder.py---
from __future__ import annotations


def add(a: int, b: int) -> int:
    \"\"\"Return the sum of two integers.\"\"\"
    return a + b
---END---

---FILE: test_adder.py---
import pytest
from adder import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-1, -2) == -3


def test_add_zero():
    assert add(0, 0) == 0


def test_add_mixed():
    assert add(-1, 5) == 4
---END---"""

_REVIEW_APPROVE = json.dumps({
    "approved": True,
    "issues": [],
    "summary": "Clean implementation with good test coverage. All functions defined and imported correctly.",
})

_COORDINATOR_APPROVE = json.dumps({
    "decision": "approve",
    "reason": "All tasks completed and approved. Code is correct and tests pass.",
    "tasks_to_rework": [],
})

_RESEARCHER_RESULT = json.dumps({
    "findings": "No research needed for this task.",
    "relevant_code": "",
    "key_points": [],
})

# Planning phase response (for two-phase coder)
_CODER_PLAN = (
    "Plan:\n"
    "1. Create adder.py with a single `add(a, b)` function\n"
    "2. Create test_adder.py with 4 pytest test cases\n"
    "3. Imports: only `pytest` and `adder`\n"
    "4. Edge cases: zero, negative numbers, mixed signs"
)

# ── Fake embedding vector ─────────────────────────────────────────── #

_FAKE_EMBEDDING = [0.01] * 768  # nomic-embed-text uses 768 dims


# ── Agent detection ───────────────────────────────────────────────── #

def _detect_agent(messages: list[dict]) -> str:
    """Detect which agent is calling based on system prompt content."""
    system = ""
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
            break

    sys_lower = system.lower()

    if "you are the coordinator" in sys_lower:
        # Distinguish scope analysis from final review
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
        if "analyze" in user_msg.lower() and "scope" in user_msg.lower():
            return "coordinator_scope"
        return "coordinator_review"
    if "you are the architect" in sys_lower:
        return "architect"
    if "you are the coder" in sys_lower:
        # Distinguish planning from coding
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
        if "before coding" in user_msg.lower() and "plan" in user_msg.lower():
            return "coder_plan"
        return "coder"
    if "you are the reviewer" in sys_lower:
        return "reviewer"
    if "you are the researcher" in sys_lower:
        return "researcher"
    return "unknown"


_RESPONSES: dict[str, str] = {
    "coordinator_scope": _SCOPE_ANALYSIS,
    "coordinator_review": _COORDINATOR_APPROVE,
    "architect": _TASK_PLAN,
    "coder": _CODER_OUTPUT,
    "coder_plan": _CODER_PLAN,
    "reviewer": _REVIEW_APPROVE,
    "researcher": _RESEARCHER_RESULT,
    "unknown": '{"status": "ok"}',
}


class MockOllamaClient:
    """Drop-in replacement for OllamaClient that returns canned responses.

    Matches the OllamaClient interface so the orchestrator and agents
    work without modification.
    """

    def __init__(self, **kwargs) -> None:
        """Accept and ignore any kwargs for compatibility."""
        self._call_count = 0

    async def chat(
        self,
        model: str,
        messages: list[dict],
        format: str | dict | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        timeout: int | None = None,
        retries: int = 2,
    ) -> OllamaResponse:
        """Return a canned response based on the detected agent."""
        self._call_count += 1
        agent = _detect_agent(messages)
        content = _RESPONSES.get(agent, _RESPONSES["unknown"])
        log.debug("MockOllama [%s] → %s (%d chars)", agent, model, len(content))
        return OllamaResponse(
            content=content,
            model=f"mock-{model}",
            total_duration=100_000,
            prompt_eval_count=10,
            eval_count=len(content) // 4,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        format: str | dict | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield the canned response as a stream of tokens."""
        agent = _detect_agent(messages)
        content = _RESPONSES.get(agent, _RESPONSES["unknown"])
        # Yield in realistic-ish chunks
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    async def generate(
        self,
        model: str,
        prompt: str,
        format: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> OllamaResponse:
        return OllamaResponse(content="{}", model=f"mock-{model}")

    async def embed(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> list[float]:
        """Return a fake embedding vector."""
        return list(_FAKE_EMBEDDING)

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return ["mock-model:latest"]

    async def close(self) -> None:
        log.debug("MockOllama closed after %d calls", self._call_count)

    @property
    def call_count(self) -> int:
        return self._call_count
