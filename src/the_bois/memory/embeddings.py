"""Vector embedding utilities — pure math, no numpy needed.

Uses Ollama's /api/embed endpoint for embedding generation and
stdlib math for cosine similarity. Because importing numpy for
dot products is like hiring a crane to hang a picture frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from the_bois.models.ollama import OllamaClient


async def embed_text(
    client: OllamaClient,
    text: str,
    model: str = "nomic-embed-text",
) -> list[float]:
    """Generate an embedding vector for the given text via Ollama.

    Truncates input to ~2000 chars to stay within nomic-embed-text's
    2K context window.  Returns an empty list on failure.
    """
    truncated = text[:2000]
    try:
        return await client.embed(truncated, model=model)
    except Exception:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_most_similar(
    query_vec: list[float],
    candidates: list[dict],
    embedding_key: str = "embedding",
    top_k: int = 3,
) -> list[tuple[dict, float]]:
    """Return the top-k most similar candidates by cosine similarity.

    Each candidate must have an `embedding_key` field containing a list[float].
    Returns list of (candidate, score) tuples, descending by score.
    """
    if not query_vec or not candidates:
        return []

    scored: list[tuple[dict, float]] = []
    for item in candidates:
        emb = item.get(embedding_key, [])
        if emb:
            score = cosine_similarity(query_vec, emb)
            scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
