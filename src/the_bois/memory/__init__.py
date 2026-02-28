"""Memory system — the bois' long-term memory.

MemoryStore is the single entry point that wraps:
- ExampleBank: gold/anti examples with embedding retrieval
- MistakeJournal: frequency-tracked anti-patterns
- EpisodeStore: full run histories for RAG

Usage:
    store = MemoryStore(config.memory)
    # ... run pipeline ...
    await store.learn_from_run(client, run_id, scope, tasks, results)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from the_bois.config import MemoryConfig
from the_bois.memory.episodes import EpisodeStore
from the_bois.memory.examples import ExampleBank
from the_bois.memory.mistakes import MistakeJournal
from the_bois.memory.scoring import Lesson, extract_lessons, score_run

if TYPE_CHECKING:
    from the_bois.models.ollama import OllamaClient


class MemoryStore:
    """Unified facade over all memory layers."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.enabled = config.enabled
        self._path = Path(config.path)
        self._path.mkdir(parents=True, exist_ok=True)

        self.examples = ExampleBank(self._path, max_examples=config.max_examples)
        self.mistakes = MistakeJournal(self._path)
        self.episodes = EpisodeStore(self._path)

    async def learn_from_run(
        self,
        client: OllamaClient,
        run_id: str,
        scope: str,
        tasks: list[dict],
        all_results: dict,
    ) -> dict:
        """Post-run learning: score, extract lessons, persist everything.

        Returns the run metrics dict for logging.
        """
        # Score the run
        run_score = score_run(all_results)
        metrics = run_score.to_dict()

        # Extract lessons
        lessons = extract_lessons(all_results)

        # Persist lessons to memory layers
        emb_model = self.config.embedding_model
        for lesson in lessons:
            if lesson.type == "gold_example":
                await self.examples.add_example(
                    client,
                    agent=lesson.agent,
                    role="gold",
                    task_description=lesson.task_description,
                    output_snippet=lesson.output_snippet,
                    embedding_model=emb_model,
                )
            elif lesson.type == "anti_example":
                await self.examples.add_example(
                    client,
                    agent=lesson.agent,
                    role="anti",
                    task_description=lesson.task_description,
                    output_snippet=lesson.output_snippet,
                    rejection_reason=lesson.rejection_reason,
                    embedding_model=emb_model,
                )
            elif lesson.type == "mistake":
                await self.mistakes.record_mistake(
                    client,
                    agent=lesson.agent,
                    pattern=lesson.rejection_reason,
                    severity=lesson.severity,
                    embedding_model=emb_model,
                )

        # Save episode
        await self.episodes.save_episode(
            client,
            run_id=run_id,
            scope=scope,
            tasks=tasks,
            results=all_results,
            metrics=metrics,
            embedding_model=emb_model,
        )

        return metrics

    @property
    def stats(self) -> dict:
        return {
            "examples": self.examples.count,
            "mistakes": self.mistakes.count,
            "episodes": self.episodes.count,
        }