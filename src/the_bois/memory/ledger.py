"""Shared conversation ledger for inter-agent communication."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class MessageType(str, Enum):
    TASK_PLAN = "task_plan"
    CODE = "code"
    REVIEW = "review"
    VALIDATION = "validation"
    RESEARCH = "research"
    DECISION = "decision"
    QUESTION = "question"
    FINAL_OUTPUT = "final_output"
    SYSTEM = "system"


@dataclass
class Message:
    from_agent: str
    to_agent: str  # agent name or "all"
    message_type: MessageType
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        data = data.copy()
        data["message_type"] = MessageType(data["message_type"])
        return cls(**data)


class Ledger:
    """Append-only message log shared across all agents."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def filter(
        self,
        from_agent: str | None = None,
        to_agent: str | None = None,
        message_type: MessageType | None = None,
        task_id: str | None = None,
    ) -> list[Message]:
        results = self._messages
        if from_agent:
            results = [m for m in results if m.from_agent == from_agent]
        if to_agent:
            results = [m for m in results if m.to_agent in (to_agent, "all")]
        if message_type:
            results = [m for m in results if m.message_type == message_type]
        if task_id:
            results = [m for m in results if m.metadata.get("task_id") == task_id]
        return results

    def get_context_for_agent(
        self,
        agent_name: str,
        max_messages: int = 20,
        task_id: str | None = None,
    ) -> list[Message]:
        """Get messages relevant to an agent, capped to most recent.

        When *task_id* is provided, only messages belonging to that task
        (or global messages with no task_id) are included.  This prevents
        the coder/reviewer from drowning in feedback from unrelated tasks.
        """
        relevant: list[Message] = []
        for m in self._messages:
            # Must be addressed to this agent (or broadcast)
            if not (m.to_agent in (agent_name, "all") or m.from_agent == agent_name):
                continue
            # Task-scope filter: keep messages for *this* task + globals
            if task_id is not None:
                msg_task = m.metadata.get("task_id")
                if msg_task is not None and msg_task != task_id:
                    continue  # belongs to a different task — skip
            relevant.append(m)
        return relevant[-max_messages:]

    def save(self, path: Path) -> None:
        data = [m.to_dict() for m in self._messages]
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self._messages = [Message.from_dict(d) for d in data]

    @property
    def count(self) -> int:
        return len(self._messages)
