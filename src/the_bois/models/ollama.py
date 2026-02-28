"""Async client wrapper for the Ollama HTTP API.

Includes retry logic, keep_alive management, and streaming support.
Because local models crash, OOM, and generally need babysitting.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# Errors worth retrying — usually transient Ollama hiccups
_RETRYABLE = (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)


@dataclass
class OllamaResponse:
    """Parsed response from Ollama."""

    content: str
    model: str
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class OllamaClient:
    """Async HTTP client for Ollama's local API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        keep_alive: str = "15m",
    ):
        self.base_url = base_url
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=base_url, timeout=float(timeout))

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
        """Send a chat completion request with retry logic.

        *format* can be:
          - ``"json"`` for free-form JSON mode
          - a dict containing a JSON Schema for grammar-constrained output
            (requires Ollama 0.5+)
          - ``None`` for plain text
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if format is not None:
            payload["format"] = format
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if num_ctx is not None:
            payload.setdefault("options", {})["num_ctx"] = num_ctx
        if num_predict is not None:
            payload.setdefault("options", {})["num_predict"] = num_predict

        req_timeout = float(timeout) if timeout else None

        for attempt in range(retries + 1):
            try:
                response = await self._client.post(
                    "/api/chat", json=payload, timeout=req_timeout
                )
                response.raise_for_status()
                data = response.json()

                return OllamaResponse(
                    content=data["message"]["content"],
                    model=data.get("model", model),
                    total_duration=data.get("total_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    eval_count=data.get("eval_count"),
                )
            except _RETRYABLE as e:
                if attempt < retries:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    log.warning(
                        "Ollama %s error (attempt %d/%d), retrying in %ds: %s",
                        model, attempt + 1, retries + 1, wait, e,
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error("Ollama %s failed after %d attempts: %s", model, retries + 1, e)
                    raise
            except httpx.HTTPStatusError as e:
                # 5xx from Ollama is usually OOM — worth a retry
                if e.response.status_code >= 500 and attempt < retries:
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        "Ollama %s returned %d (attempt %d/%d), retrying in %ds",
                        model, e.response.status_code, attempt + 1, retries + 1, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Unreachable: chat() for {model} exited retry loop")  # pragma: no cover

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        format: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content tokens as they arrive.

        Useful for progress indication and early bail-out on long generations.
        """
        import json as _json

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        if format:
            payload["format"] = format
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if num_ctx is not None:
            payload.setdefault("options", {})["num_ctx"] = num_ctx
        if num_predict is not None:
            payload.setdefault("options", {})["num_predict"] = num_predict

        req_timeout = float(timeout) if timeout else None

        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=req_timeout
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = _json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    async def generate(
        self,
        model: str,
        prompt: str,
        format: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> OllamaResponse:
        """Send a one-shot generation request."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if format:
            payload["format"] = format
        if system:
            payload["system"] = system
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        return OllamaResponse(
            content=data["response"],
            model=data.get("model", model),
            total_duration=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    async def embed(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> list[float]:
        """Generate an embedding vector for the given text."""
        payload = {"model": model, "input": text, "keep_alive": self.keep_alive}
        response = await self._client.post("/api/embed", json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        # /api/embed returns {"embeddings": [[...]]}
        embeddings = data.get("embeddings", [])
        if not embeddings:
            return []
        return embeddings[0]

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get("/")
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    async def list_models(self) -> list[str]:
        """Return names of all locally available models."""
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def close(self) -> None:
        await self._client.aclose()
