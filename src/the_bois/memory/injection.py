"""Memory injection — assembles dynamic prompt additions from all memory layers.

This is the single point where memory becomes action: before each agent
call, we query examples, mistakes, and episodes to build a context block
that gets prepended to the user prompt.

Keeps it under ~500 tokens to avoid bloating context. Quality over quantity —
one good example beats ten mediocre ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from the_bois.memory import MemoryStore
    from the_bois.models.ollama import OllamaClient


async def get_memory_context(
    client: OllamaClient,
    memory: MemoryStore,
    agent_name: str,
    task_description: str,
    scope: str = "",
    max_chars: int = 2000,
) -> str:
    """Build a memory context block to prepend to an agent's prompt.

    Assembles:
    1. Relevant gold example (if any) — "here's what good looks like"
    2. Relevant anti-example (if any) — "don't do this"
    3. Mistake warnings — "you keep doing X, stop it"
    4. Similar episode lessons (for coordinator/architect) — "last time..."

    Returns a formatted string block, or empty string if nothing relevant.
    """
    sections: list[str] = []

    # ── 1. Examples (for coder and reviewer) ──
    if agent_name in ("coder", "reviewer") and task_description:
        try:
            examples = await memory.examples.get_relevant_examples(
                client,
                agent_name,
                task_description,
                top_k=1,
            )

            gold = examples.get("gold", [])
            if gold:
                ex = gold[0]
                snippet = ex.get("output_snippet", "")[:600]
                sections.append(
                    "📗 GOOD EXAMPLE (a similar task was done well before):\n"
                    f"Task: {ex.get('task_description', '')[:200]}\n"
                    f"Output:\n{snippet}"
                )

            # Only inject anti-examples if we also have at least one gold
            # example.  All stick and no carrot just confuses the model.
            anti = examples.get("anti", [])
            if anti and gold:
                ex = anti[0]
                reason = ex.get("rejection_reason", "unknown")
                sections.append(
                    "📕 BAD EXAMPLE (a similar task failed before — avoid this):\n"
                    f"Task: {ex.get('task_description', '')[:200]}\n"
                    f"Failure reason: {reason}"
                )
        except Exception:
            pass  # Memory retrieval should never block the pipeline

    # ── 2. Mistake warnings (for all agents) ──
    try:
        warnings = memory.mistakes.get_warnings_for(agent_name, top_k=3)
        if warnings:
            # Warnings may be multi-line (pattern + fix), indent consistently
            warning_lines: list[str] = []
            for w in warnings:
                for i, line in enumerate(w.split("\n")):
                    prefix = "  ⚠ " if i == 0 else "    "
                    warning_lines.append(f"{prefix}{line}")
            warning_block = "\n".join(warning_lines)
            sections.append(
                "🚨 PAST MISTAKES (you have repeatedly made these errors — do NOT repeat them):\n"
                + warning_block
            )
    except Exception:
        pass

    # ── 3. Episode context (for coordinator and architect) ──
    if agent_name in ("coordinator", "architect") and scope:
        try:
            episodes = await memory.episodes.find_similar_episodes(
                client,
                scope,
                top_k=1,
            )
            if episodes:
                ep = episodes[0]
                summaries = ep.get("task_summaries", [])[:5]
                rate = ep.get("metrics", {}).get("approval_rate", "?")
                summary_text = "\n".join(f"  {s}" for s in summaries)
                sections.append(
                    f"📜 SIMILAR PAST PROJECT (approval rate: {rate}):\n"
                    f"Scope: {ep.get('scope_summary', '')[:200]}\n"
                    f"Tasks:\n{summary_text}"
                )
        except Exception:
            pass

    if not sections:
        return ""

    # Assemble and cap total length
    full_block = (
        "═══ MEMORY CONTEXT (learn from the past) ═══\n\n"
        + "\n\n".join(sections)
        + "\n\n═══ END MEMORY CONTEXT ═══\n\n"
    )

    if len(full_block) > max_chars:
        full_block = (
            full_block[:max_chars] + "\n... (truncated)\n═══ END MEMORY CONTEXT ═══\n\n"
        )

    return full_block
