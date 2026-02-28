"""Example Bank — stores gold and anti-examples per agent.

Gold examples: tasks that were approved on the first try.
Anti-examples: tasks that failed all review iterations.
Both are embedded for similarity retrieval so agents can learn
from past successes and screw-ups.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from the_bois.memory.embeddings import embed_text, find_most_similar

if TYPE_CHECKING:
    from the_bois.models.ollama import OllamaClient


class ExampleBank:
    """Persistent store of gold/anti examples with embedding-based retrieval."""

    def __init__(self, storage_path: Path, max_examples: int = 200) -> None:
        self._path = storage_path / "examples.json"
        self._max = max_examples
        self._examples: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._examples = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._examples = []

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._examples, indent=2))

    async def add_example(
        self,
        client: OllamaClient,
        agent: str,
        role: str,  # "gold" or "anti"
        task_description: str,
        output_snippet: str,
        rejection_reason: str = "",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        """Add a new example, embedding the task description for future retrieval."""
        embedding = await embed_text(client, task_description, model=embedding_model)

        entry = {
            "agent": agent,
            "role": role,
            "task_description": task_description,
            "output_snippet": output_snippet[:1500],  # Cap output size
            "rejection_reason": rejection_reason,
            "embedding": embedding,
            "timestamp": time.time(),
        }
        self._examples.append(entry)
        self._prune()
        self.save()

    def _prune(self) -> None:
        """Remove oldest examples when we exceed the cap."""
        if len(self._examples) > self._max:
            # Sort by timestamp ascending, keep the newest
            self._examples.sort(key=lambda e: e.get("timestamp", 0))
            self._examples = self._examples[-self._max :]

    async def get_relevant_examples(
        self,
        client: OllamaClient,
        agent: str,
        task_description: str,
        top_k: int = 2,
        embedding_model: str = "nomic-embed-text",
    ) -> dict[str, list[dict]]:
        """Retrieve the most relevant gold and anti examples for a task.

        Returns {"gold": [...], "anti": [...]}, each up to top_k entries.
        """
        if not self._examples:
            return {"gold": [], "anti": []}

        query_vec = await embed_text(client, task_description, model=embedding_model)
        if not query_vec:
            return {"gold": [], "anti": []}

        agent_examples = [e for e in self._examples if e.get("agent") == agent]
        gold = [e for e in agent_examples if e.get("role") == "gold"]
        anti = [e for e in agent_examples if e.get("role") == "anti"]

        gold_matches = find_most_similar(query_vec, gold, top_k=top_k)
        anti_matches = find_most_similar(query_vec, anti, top_k=top_k)

        return {
            "gold": [item for item, _score in gold_matches],
            "anti": [item for item, _score in anti_matches],
        }

    @property
    def count(self) -> int:
        return len(self._examples)
