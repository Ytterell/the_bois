"""Typed contracts for agent I/O — the single source of truth.

Every dict that flows between agents, the orchestrator, and the validation
pipeline has a shape defined here.  Core "objects" (files, tasks, reviews)
are dataclasses; sprawling agent-input parameter bags are TypedDicts.

Migration note: agents currently return plain dicts.  These types are
being adopted incrementally — use ``from_dict()`` at boundaries to
convert legacy dicts, and ``.to_dict()`` when serialising to JSON /
checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Required, TypedDict


# ── Enums ────────────────────────────────────────────────────────────


class Severity(str, Enum):
    """Issue severity levels used by reviewers and validators."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Decision(str, Enum):
    """Coordinator convergence decisions."""

    APPROVE = "approve"
    REWORK = "rework"
    REPLAN = "replan"


class Confidence(str, Enum):
    """Researcher confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Core dataclasses ─────────────────────────────────────────────────


@dataclass
class FileSpec:
    """A single file produced or consumed by the pipeline."""

    path: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FileSpec:
        return cls(path=data.get("path", ""), content=data.get("content", ""))


@dataclass
class ReviewIssue:
    """One issue found during review or validation."""

    severity: str  # Severity value — kept as str for LLM compat
    file: str
    description: str
    suggestion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReviewIssue:
        return cls(
            severity=data.get("severity", "major"),
            file=data.get("file", "unknown"),
            description=data.get("description", ""),
            suggestion=data.get("suggestion", ""),
        )


@dataclass
class ReviewFeedback:
    """Unified feedback shape — from reviewer, validator, or orchestrator."""

    approved: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    research_context: str = ""  # injected by orchestrator on reactive research

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ReviewFeedback:
        issues = [ReviewIssue.from_dict(i) for i in data.get("issues", [])]
        return cls(
            approved=data.get("approved", False),
            issues=issues,
            summary=data.get("summary", ""),
            research_context=data.get("research_context", ""),
        )


# ── Task specifications ──────────────────────────────────────────────


@dataclass
class SignatureSpec:
    """A required function/method signature for mechanical verification."""

    file: str
    name: str
    class_name: str = ""  # non-empty if this is a method
    params: str = ""  # e.g. "key: str, value: str"
    return_type: str = ""  # e.g. "str | None"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SignatureSpec:
        return cls(
            file=data.get("file", ""),
            name=data.get("name", ""),
            class_name=data.get("class_name", ""),
            params=data.get("params", ""),
            return_type=data.get("return_type", ""),
        )


@dataclass
class TaskSpec:
    """A single task from the architect's plan.

    The ``description`` field remains the primary spec (prose).
    Structured fields are optional, machine-checkable supplements
    derived from the description.
    """

    id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    # Phase 5 structured fields — all optional for backward compat
    required_files: list[str] = field(default_factory=list)
    required_signatures: list[SignatureSpec] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_signatures"] = [s.to_dict() for s in self.required_signatures]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TaskSpec:
        sigs = [SignatureSpec.from_dict(s) for s in data.get("required_signatures", [])]
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            required_files=data.get("required_files", []),
            required_signatures=sigs,
            acceptance_criteria=data.get("acceptance_criteria", []),
        )


# ── Agent outputs ────────────────────────────────────────────────────


@dataclass
class ScopeAnalysis:
    """Coordinator scope analysis output."""

    refined_scope: str
    needs_research: bool = False
    research_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, fallback_scope: str = "") -> ScopeAnalysis:
        return cls(
            refined_scope=data.get("refined_scope", fallback_scope),
            needs_research=data.get("needs_research", False),
            research_queries=data.get("research_queries", []),
        )


@dataclass
class ConvergenceDecision:
    """Coordinator convergence decision output."""

    decision: str  # Decision value — kept as str for LLM compat
    reason: str = ""
    tasks_to_rework: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ConvergenceDecision:
        return cls(
            decision=data.get("decision", "rework"),
            reason=data.get("reason", ""),
            tasks_to_rework=data.get("tasks_to_rework", []),
        )


@dataclass
class ResearchResult:
    """Researcher output."""

    findings: str
    key_points: list[str] = field(default_factory=list)
    relevant_code: str = ""
    pip_packages: dict[str, str] = field(default_factory=dict)
    confidence: str = "medium"  # Confidence value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ResearchResult:
        pip = data.get("pip_packages") or {}
        if not isinstance(pip, dict):
            pip = {}
        confidence = data.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        return cls(
            findings=data.get("findings", ""),
            key_points=data.get("key_points", []),
            relevant_code=data.get("relevant_code", ""),
            pip_packages=pip,
            confidence=confidence,
        )


@dataclass
class CodeOutput:
    """Coder output — parsed files + metadata."""

    files: list[FileSpec] = field(default_factory=list)
    explanation: str = ""
    already_done: bool = False
    raw_output: str = ""  # prefixed with _ in legacy dicts

    def to_dict(self) -> dict:
        d: dict = {
            "files": [f.to_dict() for f in self.files],
            "explanation": self.explanation,
            "_raw_output": self.raw_output,
        }
        if self.already_done:
            d["already_done"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CodeOutput:
        files = [FileSpec.from_dict(f) for f in data.get("files", [])]
        return cls(
            files=files,
            explanation=data.get("explanation", ""),
            already_done=data.get("already_done", False),
            raw_output=data.get("_raw_output", ""),
        )


@dataclass
class TaskResult:
    """Stored result for a completed/attempted task."""

    task: TaskSpec
    code: CodeOutput
    review: ReviewFeedback
    iterations: int = 0
    reviewer_misses: int = 0
    static_check_failures: int = 0
    fast_validation_failures: int = 0
    full_validation_failures: int = 0

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "code": self.code.to_dict(),
            "review": self.review.to_dict(),
            "iterations": self.iterations,
            "reviewer_misses": self.reviewer_misses,
            "static_check_failures": self.static_check_failures,
            "fast_validation_failures": self.fast_validation_failures,
            "full_validation_failures": self.full_validation_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskResult:
        return cls(
            task=TaskSpec.from_dict(data.get("task", {})),
            code=CodeOutput.from_dict(data.get("code", {})),
            review=ReviewFeedback.from_dict(data.get("review", {})),
            iterations=data.get("iterations", 0),
            reviewer_misses=data.get("reviewer_misses", 0),
            static_check_failures=data.get("static_check_failures", 0),
            fast_validation_failures=data.get("fast_validation_failures", 0),
            full_validation_failures=data.get("full_validation_failures", 0),
        )


# ── Agent input TypedDicts (parameter bags) ──────────────────────────


class CoderInput(TypedDict, total=False):
    """Input shape for Coder.execute()."""

    task: Required[dict]
    feedback: ReviewFeedback | dict | None
    context: dict
    failure_history: list[str]
    research_bank: dict[str, str]
    retry_hint: str
    scope: str
    plan_tasks: list[dict]
    workspace_manifest: str


class ReviewerInput(TypedDict, total=False):
    """Input shape for Reviewer.execute()."""

    task: Required[dict]
    code: Required[dict]
    context: dict
    unchanged_files: list[str]
    last_validation_error: str | list[ReviewIssue] | None
    research_bank: dict[str, str]


class ArchitectInput(TypedDict, total=False):
    """Input shape for Architect.execute()."""

    scope: Required[str]
    research: list[dict]
    feedback: str | None


class ResearcherInput(TypedDict, total=False):
    """Input shape for Researcher.execute()."""

    query: Required[str]
    max_results: int
    error_context: str
    failing_code: str
    task_description: str
    known_docs_urls: dict[str, str]


class CoordinatorDecisionInput(TypedDict, total=False):
    """Input shape for Coordinator.execute() (convergence decision)."""

    scope: Required[str]
    plan: dict
    results: dict
