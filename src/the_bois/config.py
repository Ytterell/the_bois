"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentConfig:
    model: str
    temperature: float = 0.3
    num_ctx: int | None = None
    num_predict: int | None = None
    timeout: int | None = None  # Per-agent override (seconds)


@dataclass
class OrchestrationConfig:
    max_task_iterations: int = 3
    max_global_iterations: int = 5
    context_max_messages: int = 20
    global_timeout: int = 14400  # 4 hours wall-clock max


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    keep_alive: str = "15m"


@dataclass
class WorkspaceConfig:
    path: str = "./workspace"


@dataclass
class MemoryConfig:
    enabled: bool = True
    path: str = "./memory"
    embedding_model: str = "nomic-embed-text"
    max_examples: int = 200
    max_injection_tokens: int = 500


DEFAULTS: dict[str, AgentConfig] = {
    "coordinator": AgentConfig(model="mistral:7b", temperature=0.3),
    "architect": AgentConfig(model="mistral:7b", temperature=0.4),
    "coder": AgentConfig(model="qwen2.5-coder:14b", temperature=0.2),
    "reviewer": AgentConfig(model="qwen2.5-coder:14b", temperature=0.3),
    "researcher": AgentConfig(model="phi3:3.8b", temperature=0.5),
}


@dataclass
class Config:
    agents: dict[str, AgentConfig] = field(default_factory=lambda: dict(DEFAULTS))
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults for missing values."""
    if path is None:
        path = Path("config.yaml")

    if not path.exists():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return Config()

    agents: dict[str, AgentConfig] = {}
    for name, agent_data in raw.get("agents", {}).items():
        agents[name] = AgentConfig(**agent_data)

    return Config(
        agents=agents if agents else dict(DEFAULTS),
        orchestration=OrchestrationConfig(**raw.get("orchestration", {})),
        ollama=OllamaConfig(**raw.get("ollama", {})),
        workspace=WorkspaceConfig(**raw.get("workspace", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
    )
