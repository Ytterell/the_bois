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
    ) -> None:
        """Record a mistake.  Deduplicates via embedding similarity.

        If a semantically similar mistake already exists for this agent,
        increment its frequency counter instead of adding a new entry.
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
                    # Upgrade severity if the new one is worse
                    sev_rank = {"low": 0, "medium": 1, "high": 2}
                    if sev_rank.get(severity, 1) > sev_rank.get(
                        existing.get("severity", "medium"), 1
                    ):
                        existing["severity"] = severity
                    self.save()
                    return

        # New mistake pattern
        entry = {
            "agent": agent,
            "pattern": pattern,
            "severity": severity,
            "frequency": 1,
            "embedding": new_emb,
            "first_seen": time.time(),
            "last_seen": time.time(),
        }
        self._mistakes.append(entry)
        self.save()

    def get_warnings_for(self, agent: str, top_k: int = 3) -> list[str]:
        """Return warning strings for the agent's most frequent mistakes.

        Sorted by frequency descending.  Only returns mistakes with
        frequency >= 2 (fool me once, shame on you...).
        """
        agent_mistakes = [
            m for m in self._mistakes
            if m.get("agent") == agent and m.get("frequency", 0) >= 2
        ]
        agent_mistakes.sort(key=lambda m: m.get("frequency", 0), reverse=True)

        warnings: list[str] = []
        for m in agent_mistakes[:top_k]:
            freq = m.get("frequency", 0)
            pattern = m.get("pattern", "unknown")
            sev = m.get("severity", "medium")
            warnings.append(
                f"[{sev.upper()} — seen {freq}x] {pattern}"
            )
        return warnings

    @property
    def count(self) -> int:
        return len(self._mistakes)
