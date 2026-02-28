"""Post-run scoring — the report card nobody asked for.

Analyzes completed run results to extract:
- Per-agent performance metrics (approval rate, iterations, etc.)
- Gold examples (first-try approvals = the agent nailed it)
- Anti-examples (total failures = the agent face-planted)
- Mistake patterns (repeated rejection reasons)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent."""
    tasks_attempted: int = 0
    tasks_approved: int = 0
    total_iterations: int = 0
    empty_outputs: int = 0
    already_done_count: int = 0
    circuit_breaker_hits: int = 0

    @property
    def approval_rate(self) -> float:
        if self.tasks_attempted == 0:
            return 0.0
        return self.tasks_approved / self.tasks_attempted

    @property
    def avg_iterations(self) -> float:
        if self.tasks_attempted == 0:
            return 0.0
        return self.total_iterations / self.tasks_attempted


@dataclass
class RunScore:
    """Aggregate scoring for a complete run."""
    tasks_total: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    agent_metrics: dict[str, AgentMetrics] = field(default_factory=dict)

    @property
    def approval_rate(self) -> float:
        if self.tasks_total == 0:
            return 0.0
        return self.tasks_passed / self.tasks_total

    def to_dict(self) -> dict:
        return {
            "tasks_total": self.tasks_total,
            "tasks_passed": self.tasks_passed,
            "tasks_failed": self.tasks_failed,
            "approval_rate": round(self.approval_rate, 3),
            "agent_metrics": {
                name: {
                    "tasks_attempted": m.tasks_attempted,
                    "tasks_approved": m.tasks_approved,
                    "approval_rate": round(m.approval_rate, 3),
                    "avg_iterations": round(m.avg_iterations, 2),
                    "empty_outputs": m.empty_outputs,
                    "already_done_count": m.already_done_count,
                }
                for name, m in self.agent_metrics.items()
            },
        }


@dataclass
class Lesson:
    """A single extracted lesson from a run."""
    type: str  # "gold_example", "anti_example", "mistake"
    agent: str
    task_description: str
    output_snippet: str = ""
    rejection_reason: str = ""
    severity: str = "medium"


def score_run(all_results: dict) -> RunScore:
    """Compute performance metrics for a completed run.

    Analyzes task results to produce per-agent and run-level metrics.
    Skips seed entries (task_id starting with '_').
    """
    score = RunScore()
    coder_metrics = AgentMetrics()
    reviewer_metrics = AgentMetrics()

    for task_id, result in all_results.items():
        if task_id.startswith("_"):  # Skip seed entries
            continue

        review = result.get("review", {})
        approved = review.get("approved", False)
        iterations = result.get("iterations", 0)
        code = result.get("code", {})

        score.tasks_total += 1
        if approved:
            score.tasks_passed += 1
        else:
            score.tasks_failed += 1

        # Coder metrics
        coder_metrics.tasks_attempted += 1
        coder_metrics.total_iterations += iterations
        if approved:
            coder_metrics.tasks_approved += 1

        if code.get("already_done"):
            coder_metrics.already_done_count += 1

        files = code.get("files", [])
        if not files or all(not f.get("content", "").strip() for f in files):
            coder_metrics.empty_outputs += 1

        # Reviewer gets credit for every task it reviewed
        reviewer_metrics.tasks_attempted += 1
        reviewer_metrics.total_iterations += iterations
        if approved:
            reviewer_metrics.tasks_approved += 1

    score.agent_metrics["coder"] = coder_metrics
    score.agent_metrics["reviewer"] = reviewer_metrics
    return score


def extract_lessons(all_results: dict) -> list[Lesson]:
    """Extract actionable lessons from run results.

    Identifies:
    - Gold examples: tasks approved on the first try (iterations == 1)
    - Anti-examples: tasks that failed all review iterations
    - Mistakes: rejection reasons from failed reviews
    """
    lessons: list[Lesson] = []

    for task_id, result in all_results.items():
        if task_id.startswith("_"):
            continue

        task = result.get("task", {})
        review = result.get("review", {})
        code = result.get("code", {})
        approved = review.get("approved", False)
        iterations = result.get("iterations", 0)
        task_desc = task.get("description", task.get("title", ""))

        # Build output snippet from code files
        files = code.get("files", [])
        snippet_parts = []
        for f in files[:2]:  # Max 2 files in snippet
            path = f.get("path", "?")
            content = f.get("content", "")[:500]
            snippet_parts.append(f"--- {path} ---\n{content}")
        output_snippet = "\n".join(snippet_parts)

        if approved and iterations == 1:
            # Gold example — first-try approval, the coder cooked
            lessons.append(Lesson(
                type="gold_example",
                agent="coder",
                task_description=task_desc,
                output_snippet=output_snippet,
            ))

        elif not approved:
            # Anti-example — total failure
            rejection = review.get("summary", "")
            issues = review.get("issues", [])
            issue_text = "; ".join(
                i.get("description", "") for i in issues[:3]
            )
            full_reason = f"{rejection} | Issues: {issue_text}" if issue_text else rejection

            lessons.append(Lesson(
                type="anti_example",
                agent="coder",
                task_description=task_desc,
                output_snippet=output_snippet,
                rejection_reason=full_reason,
            ))

            # Also record as a mistake pattern if there are specific issues
            for issue in issues:
                desc = issue.get("description", "")
                sev = issue.get("severity", "medium")
                if desc:
                    lessons.append(Lesson(
                        type="mistake",
                        agent="coder",
                        task_description=task_desc,
                        rejection_reason=desc,
                        severity=sev if sev in ("low", "medium", "high") else "medium",
                    ))

    return lessons
