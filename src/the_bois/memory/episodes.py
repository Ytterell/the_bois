"""Episode Store — full run histories for RAG-style retrieval.

Each completed run is saved as an episode with its scope, tasks,
outcomes, and metrics.  When a new run starts, we can find similar
past episodes by embedding the scope and doing similarity search.

Think of it as the bois' collective diary — except actually useful.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from the_bois.memory.embeddings import embed_text, find_most_similar

if TYPE_CHECKING:
    from the_bois.models.ollama import OllamaClient


class EpisodeStore:
    """Persistent store of run episodes with embedding-based retrieval."""

    def __init__(self, storage_path: Path) -> None:
        self._dir = storage_path / "episodes"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = storage_path / "episode_index.json"
        self._index: list[dict] = []  # [{run_id, scope_summary, embedding, ...}]
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._index = []

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(self._index, indent=2))

    async def save_episode(
        self,
        client: OllamaClient,
        run_id: str,
        scope: str,
        tasks: list[dict],
        results: dict,
        metrics: dict,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        """Save a complete run episode and update the index."""
        # Build a concise summary for the episode
        task_summaries = []
        for task in tasks:
            tid = task.get("id", "?")
            title = task.get("title", "?")
            result = results.get(tid, {})
            approved = result.get("review", {}).get("approved", False)
            iters = result.get("iterations", 0)
            status = "✓" if approved else "✗"
            task_summaries.append(f"{status} {tid}: {title} ({iters} iter)")

        episode = {
            "run_id": run_id,
            "scope": scope,
            "tasks": tasks,
            "task_summaries": task_summaries,
            "metrics": metrics,
            "timestamp": time.time(),
        }

        # Save full episode
        ep_path = self._dir / f"{run_id}.json"
        ep_path.write_text(json.dumps(episode, indent=2))

        # Update index with scope embedding for future retrieval
        scope_embedding = await embed_text(client, scope, model=embedding_model)
        index_entry = {
            "run_id": run_id,
            "scope_summary": scope[:300],
            "embedding": scope_embedding,
            "task_count": len(tasks),
            "approval_rate": metrics.get("approval_rate", 0),
            "timestamp": time.time(),
        }
        self._index.append(index_entry)
        self._save_index()

    async def find_similar_episodes(
        self,
        client: OllamaClient,
        scope: str,
        top_k: int = 2,
        embedding_model: str = "nomic-embed-text",
    ) -> list[dict]:
        """Find past episodes with similar project scopes.

        Returns list of episode summaries (not full data — use load_episode
        for that).
        """
        if not self._index:
            return []

        query_vec = await embed_text(client, scope, model=embedding_model)
        if not query_vec:
            return []

        matches = find_most_similar(query_vec, self._index, top_k=top_k)
        results = []
        for entry, score in matches:
            if score < 0.3:  # Don't return garbage matches
                continue
            # Load the full episode for the summary
            ep = self._load_episode(entry["run_id"])
            if ep:
                results.append({
                    "run_id": entry["run_id"],
                    "scope_summary": entry.get("scope_summary", ""),
                    "task_summaries": ep.get("task_summaries", []),
                    "metrics": ep.get("metrics", {}),
                    "similarity": round(score, 3),
                })
        return results

    def _load_episode(self, run_id: str) -> dict | None:
        ep_path = self._dir / f"{run_id}.json"
        if ep_path.exists():
            try:
                return json.loads(ep_path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    @property
    def count(self) -> int:
        return len(self._index)
