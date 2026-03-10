"""Mistake Journal — tracks recurring anti-patterns per agent.

When an agent keeps making the same mistake (e.g. outputting diffs,
forgetting imports, producing empty outputs), this module tracks
the frequency and injects warnings into future prompts.

Fuzzy dedup via embedding similarity > 0.85 so "forgot to close file"
and "didn't close the file handle" collapse into one pattern.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from the_bois.memory.embeddings import cosine_similarity, embed_text

if TYPE_CHECKING:
    from the_bois.models.ollama import OllamaClient

# Similarity threshold for treating two mistake descriptions as the same
DEDUP_THRESHOLD = 0.85

# Hard cap on stored mistakes — beyond this, prune lowest-value entries
_MAX_MISTAKES = 50


def _severity_for_frequency(freq: int) -> str:
    """Auto-escalate severity based on how often a mistake recurs."""
    if freq >= 6:
        return "high"
    if freq >= 3:
        return "medium"
    return "low"


class MistakeJournal:
    """Persistent store of agent anti-patterns with frequency tracking."""

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path / "mistakes.json"
        self._mistakes: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._mistakes = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._mistakes = []

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._mistakes, indent=2))

    async def record_mistake(
        self,
        client: OllamaClient,
        agent: str,
        pattern: str,
        severity: str = "medium",  # "low", "medium", "high"
        embedding_model: str = "nomic-embed-text",
        root_cause: str = "",
        fix_approach: str = "",
    ) -> None:
        """Record a mistake.  Deduplicates via embedding similarity.

        If a semantically similar mistake already exists for this agent,
        increment its frequency counter instead of adding a new entry.
        On dedup, root_cause and fix_approach are updated if non-empty
        (latest info wins — it's usually more specific).
        """
        new_emb = await embed_text(client, pattern, model=embedding_model)

        # Check for existing similar mistakes for this agent
        agent_mistakes = [m for m in self._mistakes if m.get("agent") == agent]
        for existing in agent_mistakes:
            existing_emb = existing.get("embedding", [])
            if existing_emb and new_emb:
                sim = cosine_similarity(new_emb, existing_emb)
                if sim >= DEDUP_THRESHOLD:
                    existing["frequency"] = existing.get("frequency", 1) + 1
                    existing["last_seen"] = time.time()
                    # Auto-escalate severity by frequency
                    existing["severity"] = _severity_for_frequency(
                        existing["frequency"]
                    )
                    # Update structured fields if new info is provided
                    if root_cause:
                        existing["root_cause"] = root_cause
                    if fix_approach:
                        existing["fix_approach"] = fix_approach
                    self.save()
                    return

        # New mistake pattern
        entry = {
            "agent": agent,
            "pattern": pattern,
            "severity": _severity_for_frequency(1),
            "frequency": 1,
            "embedding": new_emb,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "root_cause": root_cause,
            "fix_approach": fix_approach,
        }
        self._mistakes.append(entry)
        self._prune()
        self.save()

    def _prune(self) -> None:
        """Evict lowest-value mistakes when we exceed the cap.

        Value = frequency * severity_weight.  Keeps the most recurring,
        most severe patterns and tosses the one-off noise.
        """
        if len(self._mistakes) <= _MAX_MISTAKES:
            return
        sev_weight = {"low": 1, "medium": 2, "high": 3}
        self._mistakes.sort(
            key=lambda m: (
                m.get("frequency", 1) * sev_weight.get(m.get("severity", "low"), 1)
            ),
            reverse=True,
        )
        self._mistakes = self._mistakes[:_MAX_MISTAKES]

    def get_warnings_for(self, agent: str, top_k: int = 3) -> list[str]:
        """Return warning strings for the agent's most frequent mistakes.

        Sorted by frequency descending.  Only returns mistakes with
        frequency >= 2 (fool me once, shame on you...).

        Includes root_cause and fix_approach when available for
        actionable guidance — not just "what" but "why" and "how to fix".
        """
        agent_mistakes = [
            m
            for m in self._mistakes
            if m.get("agent") == agent and m.get("frequency", 0) >= 2
        ]
        agent_mistakes.sort(key=lambda m: m.get("frequency", 0), reverse=True)

        warnings: list[str] = []
        for m in agent_mistakes[:top_k]:
            freq = m.get("frequency", 0)
            pattern = m.get("pattern", "unknown")
            sev = m.get("severity", "medium")
            root_cause = m.get("root_cause", "")
            fix_approach = m.get("fix_approach", "")

            parts = [f"[{sev.upper()} — seen {freq}x] {pattern}"]
            if fix_approach:
                parts.append(f"  Fix: {fix_approach}")
            elif root_cause:
                # Only show root_cause if no fix_approach (fix is more useful)
                parts.append(f"  Cause: {root_cause}")
            warnings.append("\n".join(parts))
        return warnings

    @property
    def count(self) -> int:
        return len(self._mistakes)
