"""Base agent class — all agents inherit from this."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from the_bois.config import AgentConfig
from the_bois.memory.ledger import Ledger, Message, MessageType
from the_bois.models.ollama import OllamaClient
from the_bois.utils import estimate_tokens

if TYPE_CHECKING:
    from the_bois.memory import MemoryStore

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all agents in the system."""

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        client: OllamaClient,
        ledger: Ledger,
        memory: MemoryStore | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.client = client
        self.ledger = ledger
        self.memory = memory
        # Orchestrator can override temperature per-call (e.g. ramp on retry)
        self._temperature_override: float | None = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define the agent's role and behaviour."""
        ...

    def _budget_context(
        self,
        system_text: str,
        prompt_text: str,
    ) -> tuple[int, int]:
        """Calculate token budgets for ledger and memory injection.

        Returns (ledger_budget, memory_budget) in estimated tokens.
        The model's context window is a precious resource — treat it like
        VRAM at 2am: every byte matters.
        """
        ctx_limit = self.config.num_ctx or 4096
        predict_budget = self.config.num_predict or 2048
        input_budget = ctx_limit - predict_budget

        system_tokens = estimate_tokens(system_text)
        prompt_tokens = estimate_tokens(prompt_text)
        must_have = system_tokens + prompt_tokens

        remaining = max(input_budget - must_have, 0)

        # Memory gets 35% of remaining, ledger gets 65%
        memory_budget = max(int(remaining * 0.35), 100)
        ledger_budget = remaining - memory_budget

        if remaining < 200:
            log.warning(
                "%s: tight context budget (%d tokens remaining after system+prompt). "
                "Ledger and memory will be minimal.",
                self.name, remaining,
            )

        return ledger_budget, memory_budget

    async def think(
        self,
        prompt: str,
        json_mode: bool = False,
        json_schema: dict | None = None,
        task_description: str = "",
        scope: str = "",
        task_id: str | None = None,
    ) -> str:
        """Build context from the ledger + memory, send to LLM, return response text.

        Now with token budget management so we don't blow the context window
        and have the model forget its own name.
        """
        system_text = self.system_prompt
        ledger_budget, memory_budget = self._budget_context(system_text, prompt)

        messages: list[dict] = [
            {"role": "system", "content": system_text},
        ]

        # ── Inject ledger history within budget (newest first) ──
        ledger_messages = self.ledger.get_context_for_agent(
            self.name, task_id=task_id,
        )
        used_tokens = 0
        budgeted_msgs: list[tuple[str, str]] = []  # (role, content)

        for msg in reversed(ledger_messages):
            content = (
                f"[{msg.from_agent} → {msg.to_agent}] "
                f"({msg.message_type.value})\n{msg.content}"
            )
            msg_tokens = estimate_tokens(content)
            if used_tokens + msg_tokens > ledger_budget:
                break
            role = "assistant" if msg.from_agent == self.name else "user"
            budgeted_msgs.insert(0, (role, content))
            used_tokens += msg_tokens

        for role, content in budgeted_msgs:
            messages.append({"role": role, "content": content})

        # ── Inject memory context within budget ──
        memory_prefix = ""
        if self.memory and self.memory.enabled:
            try:
                from the_bois.memory.injection import get_memory_context

                memory_prefix = await get_memory_context(
                    client=self.client,
                    memory=self.memory,
                    agent_name=self.name,
                    task_description=task_description,
                    scope=scope,
                    max_chars=memory_budget * 4,  # convert tokens back to ~chars
                )
            except Exception:
                pass  # Memory should never break the pipeline

        messages.append({"role": "user", "content": memory_prefix + prompt})

        temperature = self._temperature_override or self.config.temperature

        # json_schema takes precedence over json_mode
        fmt: str | dict | None = None
        if json_schema:
            fmt = json_schema
        elif json_mode:
            fmt = "json"

        response = await self.client.chat(
            model=self.config.model,
            messages=messages,
            format=fmt,
            temperature=temperature,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
            timeout=self.config.timeout,
        )
        return response.content

    async def think_stream(
        self,
        prompt: str,
        task_description: str = "",
        task_id: str | None = None,
    ) -> str:
        """Like think(), but streams tokens and shows a live progress indicator.

        Only for non-JSON agents (coder). Returns the complete accumulated
        response text, same as think() for downstream compatibility.
        """
        from rich.live import Live
        from rich.text import Text as RichText

        system_text = self.system_prompt
        ledger_budget, memory_budget = self._budget_context(system_text, prompt)

        messages: list[dict] = [
            {"role": "system", "content": system_text},
        ]

        # Ledger (same logic as think)
        ledger_messages = self.ledger.get_context_for_agent(
            self.name, task_id=task_id,
        )
        used_tokens = 0
        budgeted_msgs: list[tuple[str, str]] = []
        for msg in reversed(ledger_messages):
            content = (
                f"[{msg.from_agent} → {msg.to_agent}] "
                f"({msg.message_type.value})\n{msg.content}"
            )
            msg_tokens = estimate_tokens(content)
            if used_tokens + msg_tokens > ledger_budget:
                break
            role = "assistant" if msg.from_agent == self.name else "user"
            budgeted_msgs.insert(0, (role, content))
            used_tokens += msg_tokens
        for role, content in budgeted_msgs:
            messages.append({"role": role, "content": content})

        # Memory
        memory_prefix = ""
        if self.memory and self.memory.enabled:
            try:
                from the_bois.memory.injection import get_memory_context
                memory_prefix = await get_memory_context(
                    client=self.client,
                    memory=self.memory,
                    agent_name=self.name,
                    task_description=task_description,
                    max_chars=memory_budget * 4,
                )
            except Exception:
                pass

        messages.append({"role": "user", "content": memory_prefix + prompt})

        temperature = self._temperature_override or self.config.temperature

        # Stream with live progress
        accumulated: list[str] = []
        token_count = 0

        with Live(
            RichText("  ✏️  Generating... 0 tokens", style="dim"),
            refresh_per_second=4,
            transient=True,
        ) as live:
            async for token in self.client.chat_stream(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                num_ctx=self.config.num_ctx,
                num_predict=self.config.num_predict,
                timeout=self.config.timeout,
            ):
                accumulated.append(token)
                token_count += 1
                if token_count % 10 == 0:  # update every 10 tokens
                    chars = sum(len(t) for t in accumulated)
                    live.update(RichText(
                        f"  ✏️  Generating... {token_count} tokens ({chars} chars)",
                        style="dim",
                    ))

        return "".join(accumulated)

    def post_message(
        self,
        to_agent: str,
        message_type: MessageType,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        """Post a message to the shared ledger."""
        message = Message(
            from_agent=self.name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        self.ledger.append(message)
        return message

    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        """Run the agent's primary function. Subclasses implement this."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} model={self.config.model}>"
