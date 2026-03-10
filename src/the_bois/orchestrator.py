"""Main orchestration loop — coordinates agents to solve a given scope."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from the_bois.agents.architect import ArchitectAgent
from the_bois.agents.coder import CoderAgent
from the_bois.agents.coordinator import CoordinatorAgent
from the_bois.agents.researcher import ResearcherAgent
from the_bois.agents.reviewer import ReviewerAgent
from the_bois.config import Config
from the_bois.contracts import (
    CodeOutput,
    CoderInput,
    CoordinatorDecisionInput,
    FileSpec,
    ResearcherInput,
    ReviewerInput,
    ReviewFeedback,
    ReviewIssue,
    TaskResult,
    TaskSpec,
)
from the_bois.memory import MemoryStore
from the_bois.memory.ledger import Ledger, Message, MessageType
from the_bois.models.ollama import OllamaClient
from the_bois.tools.repair import auto_repair
from the_bois.tools.static_check import (
    static_check_files,
    self_verify,
    verify_signatures,
)
from the_bois.tools.context import chunk_files_for_context
from the_bois.tools.error_taxonomy import (
    ClassifiedError,
    classify_errors,
    classify_validation_result,
    get_dominant_strategy,
    build_taxonomy_retry_hint,
    ErrorCategory,
)
from the_bois.tools.validator import validate_code, validate_fast, validate_full
from the_bois.tools.workspace import Workspace
from the_bois.tools.tracing import RunTracer

console = Console()
log = logging.getLogger(__name__)

# Agent display styles
AGENT_STYLE = {
    "coordinator": ("bold yellow", "🎯"),
    "architect": ("bold blue", "📐"),
    "coder": ("bold green", "💻"),
    "reviewer": ("bold red", "🔍"),
    "researcher": ("bold magenta", "🔎"),
}


def agent_header(agent: str, message: str) -> Panel:
    """Pretty header for agent activity."""
    style, icon = AGENT_STYLE.get(agent, ("bold white", "⚙"))
    return Panel(
        Text(message, style="white"),
        title=f"{icon} {agent.upper()}",
        border_style=style,
        expand=False,
    )


class Orchestrator:
    """Drives the multi-agent collaboration loop."""

    def __init__(self, config: Config, client=None, run_id: str | None = None) -> None:
        self.config = config
        self.ledger = Ledger()
        self.client = client or OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
            keep_alive=config.ollama.keep_alive,
        )
        # Each run gets its own timestamped subfolder
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = Path(config.workspace.path) / f"run_{self.run_id}"
        self.workspace = Workspace(run_path)
        self.tracer = RunTracer(run_path / "trace.jsonl")
        self.agents: dict = {}
        self.research_bank: dict[
            str, str
        ] = {}  # query → findings (persists entire run)
        self.research_cache_hits: dict[str, int] = {}  # query → times served from cache
        self.known_docs_urls: dict[str, str] = {}  # package_name → docs URL (persists)
        self.reviewer_misses: int = 0  # times reviewer approved but validation rejected

        # Sandboxed deps directory — persists across tasks within a run
        self.deps_dir: str = tempfile.mkdtemp(prefix="the_bois_deps_")
        log.info("Deps directory: %s", self.deps_dir)

        # Metrics tracking
        self.static_check_failures: int = 0  # static analysis failures
        self.fast_validation_failures: int = 0  # syntax/import failures
        self.full_validation_failures: int = 0  # test failures
        self.total_coder_iterations: int = 0  # total coder attempts across all tasks

        # Timing tracking
        self.timing: dict[str, float] = {}  # agent_name -> total seconds

        # Memory system
        self.memory: MemoryStore | None = None
        if config.memory.enabled:
            self.memory = MemoryStore(config.memory)

    async def _timed_await(self, agent_name: str, coro) -> Any:
        """Wrap an await with timing."""
        import time

        start = time.perf_counter()
        try:
            return await coro
        finally:
            elapsed = time.perf_counter() - start
            self.timing[agent_name] = self.timing.get(agent_name, 0) + elapsed

    def _init_agents(self) -> None:
        """Instantiate all agents with their configs."""
        max_msgs = self.config.orchestration.context_max_messages
        args = lambda name: (
            name,
            self.config.agents[name],
            self.client,
            self.ledger,
            self.memory,
        )

        self.agents = {
            "coordinator": CoordinatorAgent(
                *args("coordinator"), context_max_messages=max_msgs
            ),
            "architect": ArchitectAgent(
                *args("architect"), context_max_messages=max_msgs
            ),
            "coder": CoderAgent(*args("coder"), context_max_messages=max_msgs),
            "reviewer": ReviewerAgent(*args("reviewer"), context_max_messages=max_msgs),
            "researcher": ResearcherAgent(
                *args("researcher"), context_max_messages=max_msgs
            ),
        }

    async def run(self, scope: str, seed_files: dict[str, str] | None = None) -> None:
        """Main entry — takes a problem scope and runs the agent loop.

        Args:
            scope: The problem description.
            seed_files: Optional dict of {path: content} from a previous run
                        to resume/iterate on.
        """
        if not await self.client.health_check():
            console.print(
                "[bold red]✗ Cannot connect to Ollama. Is it running?[/bold red]"
            )
            return

        self._init_agents()

        try:
            await self._execute_pipeline(scope, seed_files=seed_files)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted — saving current state...[/yellow]")
        finally:
            self._save_state()
            await self.client.close()
            # Cleanup deps directory
            if self.deps_dir and Path(self.deps_dir).exists():
                shutil.rmtree(self.deps_dir, ignore_errors=True)
                log.info("Cleaned up deps directory: %s", self.deps_dir)

    async def _execute_pipeline(
        self,
        scope: str,
        seed_files: dict[str, str] | None = None,
    ) -> None:
        """The full Coordinator → Researcher → Architect → Coder ↔ Reviewer → Coordinator pipeline."""

        # ── Build seed context from previous run (if any) ──
        seed_context = ""
        seed_results: dict = {}
        if seed_files:
            seed_file_list = [{"path": p, "content": c} for p, c in seed_files.items()]
            seed_results["_seed"] = TaskResult(
                task=TaskSpec(
                    id="_seed",
                    title="Previous run output",
                    description="",
                ),
                code=CodeOutput(
                    files=[FileSpec(path=p, content=c) for p, c in seed_files.items()],
                ),
                review=ReviewFeedback(
                    approved=True,
                    summary="Seed from previous run.",
                ),
            ).to_dict()
            seed_context = (
                "\n\nEXISTING CODE FROM A PREVIOUS ATTEMPT (focus on what "
                "needs fixing, completing, or improving — do NOT redo what already works):\n"
            )
            for p, c in seed_files.items():
                seed_context += f"\n--- {p} ---\n{c}\n"

        # ── Step 0a: Coordinator analyzes scope ──
        console.print(agent_header("coordinator", "Analyzing scope for clarity..."))
        scope_with_seed = scope + seed_context if seed_context else scope
        with self.tracer.span("scope_analysis"):
            scope_analysis = await self._timed_await(
                "coordinator", self.agents["coordinator"].analyze_scope(scope_with_seed)
            )
        refined_scope = scope_analysis.get("refined_scope", scope)
        needs_research = scope_analysis.get("needs_research", False)
        research_queries = scope_analysis.get("research_queries", [])

        log.info("Refined scope: %s", refined_scope[:500])
        if needs_research:
            console.print(
                f"  [magenta]Research needed: {len(research_queries)} query(ies)[/magenta]"
            )
        log.info("Needs research: %s, queries: %s", needs_research, research_queries)
        console.print()

        # ── Step 0b: Researcher gathers context (if needed) ──
        research_findings: list[dict] = []
        if needs_research and research_queries:
            for query in research_queries:
                console.print(agent_header("researcher", f"Researching: {query}"))
                try:
                    with self.tracer.span("research", query=query[:100]):
                        result = await self._timed_await(
                            "researcher",
                            self.agents["researcher"].execute({"query": query}),
                        )
                    research_findings.append({"query": query, **result})
                    confidence = result.get("confidence", "medium")
                    log.info(
                        "Research findings for '%s' (%s confidence): %s",
                        query,
                        confidence,
                        result.get("findings", "")[:300],
                    )

                    # Persist pip_packages mapping
                    pip_pkgs = result.get("pip_packages") or {}
                    if isinstance(pip_pkgs, dict):
                        for import_name, pip_name in pip_pkgs.items():
                            if import_name and pip_name:
                                self.known_docs_urls.setdefault(import_name, "")
                                log.info(
                                    "pip mapping: import %s → pip install %s",
                                    import_name,
                                    pip_name,
                                )

                    # D: Only cache medium/high confidence results
                    if confidence == "low":
                        log.warning(
                            "Low-confidence research for '%s' — not caching",
                            query[:80],
                        )
                        continue

                    # Store in persistent research bank for coder access
                    block = ""
                    findings = result.get("findings", "")
                    code_ref = result.get("relevant_code", "")
                    key_points = result.get("key_points", [])
                    if findings:
                        block += findings[:1500]
                    if key_points:
                        block += "\nKey points:\n" + "\n".join(
                            f"  • {p}" for p in key_points[:5]
                        )
                    if code_ref:
                        block += f"\nCode reference:\n{code_ref[:1000]}"
                    # Include pip_packages mapping if available
                    if pip_pkgs:
                        block += "\npip packages: " + ", ".join(
                            f"{k} → pip install {v}" for k, v in pip_pkgs.items()
                        )
                    if block:
                        self.research_bank[query] = block

                except Exception as e:
                    console.print(f"  [yellow]⚠ Research failed: {e}[/yellow]")

            if self.research_bank:
                log.info("Research bank: %d topic(s) cached", len(self.research_bank))
            console.print()

        # Store scope + plan for coder context injection
        self._current_scope = refined_scope
        self._current_plan_tasks: list[dict] = []

        # ── Step 1: Architect decomposes the problem ──
        console.print(
            agent_header("architect", "Analyzing scope and creating task plan...")
        )
        architect_scope = (
            refined_scope + seed_context if seed_context else refined_scope
        )
        with self.tracer.span("architect"):
            plan = await self._timed_await(
                "architect",
                self.agents["architect"].execute(
                    {"scope": architect_scope, "research": research_findings}
                ),
            )
        tasks = plan.get("tasks", [])

        if not tasks:
            console.print("[bold red]Architect produced no tasks. Aborting.[/bold red]")
            return

        self._current_plan_tasks = tasks

        console.print(f"\n[bold blue]📋 Plan: {len(tasks)} task(s)[/bold blue]")
        for t in tasks:
            console.print(f"  [blue]• {t['id']}:[/blue] {t['title']}")
        console.print()

        # ── Step 2: Execute tasks with review loops ──
        all_results: dict = dict(seed_results)  # Pre-load seed if any
        max_global = self.config.orchestration.max_global_iterations
        wall_clock_start = time.monotonic()
        wall_clock_limit = self.config.orchestration.global_timeout

        for global_iter in range(max_global):
            # ── Wall-clock timeout check ──
            elapsed = time.monotonic() - wall_clock_start
            if elapsed >= wall_clock_limit:
                console.print(
                    f"[bold yellow]⚠ Wall-clock timeout ({wall_clock_limit}s) hit "
                    f"after {elapsed:.0f}s. Saving best effort.[/bold yellow]\n"
                )
                break

            console.print(
                Rule(f"[bold]Global Iteration {global_iter + 1}/{max_global}[/bold]")
            )

            failed_deps = False
            for task in tasks:
                # Check if this task's dependencies were all approved
                deps = task.get("dependencies", [])
                unmet = [
                    d
                    for d in deps
                    if d in all_results
                    and not all_results[d].get("review", {}).get("approved", False)
                ]
                if unmet:
                    # Merge failed dependency descriptions into this task
                    merged_desc = task["description"]
                    for dep_id in unmet:
                        dep_task = all_results[dep_id].get("task", {})
                        merged_desc = (
                            f"NOTE: A prerequisite task failed. You must ALSO implement "
                            f"the following:\n"
                            f"  Title: {dep_task.get('title', dep_id)}\n"
                            f"  Description: {dep_task.get('description', 'N/A')}\n\n"
                            + merged_desc
                        )
                    task = {**task, "description": merged_desc}
                    log.info("Merging failed dep(s) %s into %s", unmet, task["id"])

                # Build context: only relevant files for this task
                filtered_context = self._build_coder_context(
                    all_results, current_task=task
                )

                result = await self._execute_task(task, filtered_context)
                all_results[task["id"]] = result

                # Track total iterations across all tasks
                self.total_coder_iterations += result.get("iterations", 1)

                # Auto-checkpoint after approved tasks — overnight insurance
                if result.get("review", {}).get("approved", False):
                    self._checkpoint(all_results)
                else:
                    console.print(
                        f"  [yellow]⚠ Task {task['id']} failed all attempts[/yellow]"
                    )

            # ── Step 3: Coordinator reviews everything ──
            console.print(agent_header("coordinator", "Reviewing overall progress..."))
            with self.tracer.span("coordinator_decision", global_iter=global_iter):
                decision = await self._timed_await(
                    "coordinator",
                    self.agents["coordinator"].execute(
                        {"scope": refined_scope, "plan": plan, "results": all_results}
                    ),
                )

            dec = decision.get("decision", "approve")
            reason = decision.get("reason", "No reason given.")
            console.print(f"  [dim]Decision:[/dim] [bold]{dec}[/bold] — {reason}\n")

            if dec == "approve":
                console.print(
                    "[bold green]✅ The bois are satisfied. Work complete.[/bold green]\n"
                )

                # Display run metrics
                console.print("\n[bold]📊 Run Metrics:[/bold]")
                console.print(
                    f"  Total coder iterations: {self.total_coder_iterations}"
                )
                console.print(f"  Static check failures: {self.static_check_failures}")
                console.print(
                    f"  Fast validation failures: {self.fast_validation_failures}"
                )
                console.print(
                    f"  Full validation failures: {self.full_validation_failures}"
                )
                console.print(f"  Reviewer misses: {self.reviewer_misses}")

                # Display timing breakdown
                if self.timing:
                    total_time = sum(self.timing.values())
                    console.print("\n[bold]⏱️ Timing Breakdown:[/bold]")
                    for agent, secs in sorted(self.timing.items(), key=lambda x: -x[1]):
                        pct = (secs / total_time * 100) if total_time > 0 else 0
                        mins, secs_remain = divmod(int(secs), 60)
                        if mins:
                            console.print(
                                f"  {agent}: {mins}m {secs_remain}s ({pct:.1f}%)"
                            )
                        else:
                            console.print(f"  {agent}: {secs_remain}s ({pct:.1f}%)")
                    console.print(f"  Total: {int(total_time)}s")

                console.print()

                break

            elif dec == "rework":
                rework_ids = set(decision.get("tasks_to_rework", []))
                tasks = [t for t in plan.get("tasks", []) if t["id"] in rework_ids]
                if not tasks:
                    console.print(
                        "[yellow]Coordinator requested rework but listed no valid task IDs. Approving.[/yellow]"
                    )
                    break
                console.print(
                    f"[yellow]↻ Reworking {len(tasks)} task(s): {', '.join(t['id'] for t in tasks)}[/yellow]\n"
                )

            elif dec == "replan":
                console.print("[yellow]↻ Coordinator wants a full replan...[/yellow]\n")
                console.print(
                    agent_header("architect", "Re-planning based on feedback...")
                )
                plan = await self._timed_await(
                    "architect",
                    self.agents["architect"].execute(
                        {
                            "scope": refined_scope,
                            "feedback": reason,
                            "research": research_findings,
                        }
                    ),
                )
                tasks = plan.get("tasks", [])
                if not tasks:
                    console.print(
                        "[bold red]Architect produced no tasks on replan. Stopping.[/bold red]"
                    )
                    break
                # Preserve approved results across replans — don't nuke
                # work that already passed review
                all_results = {
                    tid: res
                    for tid, res in all_results.items()
                    if res.get("review", {}).get("approved", False)
                }

        else:
            console.print(
                f"[bold yellow]⚠ Hit max global iterations ({max_global}). "
                f"Outputting best effort.[/bold yellow]\n"
            )

        # ── Step 4: Validate before saving ──
        all_results = self._validate_outputs(all_results)

        # ── Step 4.5: Generate requirements.txt from all installed deps ──
        self._generate_requirements(all_results)

        # ── Step 5: Save results ──
        await self._finalize(all_results)

        # ── Step 6: Learn from this run ──
        await self._learn(refined_scope, tasks, all_results)

    def _build_coder_context(
        self,
        all_results: dict,
        current_task: dict | None = None,
    ) -> dict:
        """Build deduplicated file context for the coder and reviewer.

        Collapses approved results into a set of files, deduplicated by path
        (latest version wins).  When *current_task* is provided, only files
        from dependency tasks or files explicitly mentioned in the task
        description are included — this keeps the coder's context focused
        and avoids wasting tokens on irrelevant code.

        Falls back to all files when the task has no dependencies and no
        file mentions (e.g. the first task in a plan).
        """
        files_by_path: dict[str, dict] = {}
        context: dict = {}

        # Collect relevant task IDs based on dependencies
        dep_ids: set[str] | None = None
        mentioned_paths: set[str] | None = None
        if current_task:
            deps = set(current_task.get("dependencies", []))
            desc = current_task.get("description", "")
            # Extract file paths mentioned in the task description
            path_mentions = set(
                re.findall(r"[\w./]+\.(?:py|js|ts|json|yaml|yml|toml|cfg|txt|md)", desc)
            )
            if deps or path_mentions:
                dep_ids = deps
                mentioned_paths = path_mentions

        for task_id, result in all_results.items():
            approved = result.get("review", {}).get("approved", False)
            if approved:
                for f in result.get("code", {}).get("files", []):
                    path = f.get("path", "")
                    if path and f.get("content", "").strip():
                        files_by_path[path] = f  # last writer wins
            else:
                # Only pass the task plan, not the broken code
                context[task_id] = {
                    "task": result.get("task", {}),
                    "code": {"files": []},
                    "review": result.get("review", {}),
                    "_failed": True,
                }

        # Filter files when we have dependency/mention info
        if dep_ids is not None or mentioned_paths is not None:
            dep_ids = dep_ids or set()
            mentioned_paths = mentioned_paths or set()

            # Collect paths from dependency tasks
            dep_paths: set[str] = set()
            for tid in dep_ids:
                res = all_results.get(tid, {})
                for f in res.get("code", {}).get("files", []):
                    dep_paths.add(f.get("path", ""))

            filtered: dict[str, dict] = {}
            for path, fdict in files_by_path.items():
                if path in dep_paths or any(m in path for m in mentioned_paths):
                    filtered[path] = fdict

            # If filtering leaves nothing (no matches), fall back to all
            if filtered:
                files_by_path = filtered

        # Single entry with all unique files — coder/reviewer see each
        # file exactly once regardless of how many tasks touched it.
        if files_by_path:
            files_list = list(files_by_path.values())

            # Chunk large files based on task description for token optimization
            if current_task:
                task_desc = current_task.get("description", "")
                files_list = chunk_files_for_context(files_list, task_desc)

            context["_codebase"] = {
                "code": {"files": files_list},
                "review": {"approved": True},
            }

        return context

    @staticmethod
    def _existing_file_hashes(existing_results: dict) -> dict[str, str]:
        """Build path → md5 map from all files in existing approved results."""
        hashes: dict[str, str] = {}
        for result in existing_results.values():
            if not result.get("review", {}).get("approved", False):
                continue
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content.strip():
                    hashes[path] = hashlib.md5(content.encode()).hexdigest()
        return hashes

    def _strip_unchanged_files(
        self,
        code_result: dict,
        existing_results: dict,
    ) -> tuple[dict, list[str]]:
        """Remove files from coder output that are identical to existing versions.

        Returns:
            (filtered_code_result, list_of_unchanged_paths)

        The original code_result is NOT mutated — a new dict is returned
        with only changed/new files. The unchanged paths list lets the
        reviewer know what was omitted.
        """
        existing_hashes = self._existing_file_hashes(existing_results)
        if not existing_hashes:
            return code_result, []  # first task — nothing to compare

        changed_files: list[dict] = []
        unchanged_paths: list[str] = []

        for f in code_result.get("files", []):
            path = f.get("path", "")
            content = f.get("content", "")
            if not path:
                continue
            new_hash = hashlib.md5(content.encode()).hexdigest()
            if path in existing_hashes and new_hash == existing_hashes[path]:
                unchanged_paths.append(path)
            else:
                changed_files.append(f)

        if unchanged_paths:
            log.info(
                "Stripped %d unchanged file(s): %s",
                len(unchanged_paths),
                unchanged_paths,
            )

        filtered = {**code_result, "files": changed_files}
        return filtered, unchanged_paths

    async def _execute_task(self, task: dict, existing_results: dict) -> dict:
        """Run the Coder → Reviewer loop for a single task."""
        task_id = task["id"]
        max_iters = self.config.orchestration.max_task_iterations
        feedback = None
        code_result: dict = {}
        review_result: dict = {}
        best_code_result: dict = {}  # preserve the best code across iterations
        iteration = 0
        prev_content_hash: str = ""
        failure_history: list[str] = []  # per-task compressed failure log
        last_validation_error: str | None = None  # for reviewer awareness
        coder_agent: CoderAgent = self.agents["coder"]
        base_temp = coder_agent.config.temperature
        research_requests_this_task: int = 0  # cap coder-initiated research
        retry_hint: str = ""  # escalating hints for repeated failure patterns
        last_classified: list = []  # ClassifiedError list from last failure

        for iteration in range(max_iters):
            iter_label = f"(attempt {iteration + 1}/{max_iters})"

            # ── J: Detect repeated failure patterns and inject retry_hint ──
            # Layer 1: taxonomy-based hint from the most recent failure
            taxonomy_hint = build_taxonomy_retry_hint(last_classified)
            # Layer 2: pattern-based escalation from repeated failures
            pattern_hint = self._build_retry_hint(failure_history)
            # Combine — taxonomy is more specific, pattern catches repetition
            hint_parts = [h for h in (taxonomy_hint, pattern_hint) if h]
            retry_hint = "\n".join(hint_parts)

            # ── P: Taxonomy-driven temperature adjustment ──
            if iteration > 0 and last_classified:
                strategy = get_dominant_strategy(last_classified)
                if strategy.temperature_delta != 0:
                    adjusted = min(
                        max(base_temp + strategy.temperature_delta * iteration, 0.1),
                        1.0,
                    )
                    coder_agent._temperature_override = adjusted
                    log.info(
                        "Temperature adjusted to %.2f (delta=%.2f) for %s",
                        adjusted,
                        strategy.temperature_delta,
                        task_id,
                    )
                else:
                    coder_agent._temperature_override = None
                    log.info("Temperature held at base (no delta) for %s", task_id)
            else:
                coder_agent._temperature_override = None

            # Coder writes
            console.print(agent_header("coder", f"Task: {task['title']} {iter_label}"))

            # Build compact workspace manifest so coder knows what files exist
            workspace_manifest = self._build_workspace_manifest(existing_results)

            with self.tracer.span("coder", task_id=task_id, iteration=iteration):
                code_result = await self._timed_await(
                    "coder",
                    coder_agent.execute(
                        {
                            "task": task,
                            "feedback": feedback,
                            "context": existing_results,
                            "failure_history": failure_history,
                            "research_bank": self.research_bank,
                            "retry_hint": retry_hint,
                            "scope": self._current_scope,
                            "plan_tasks": self._current_plan_tasks,
                            "workspace_manifest": workspace_manifest,
                        }
                    ),
                )

            # Handle ALREADY_DONE signal — coder says existing code covers this
            if code_result.get("already_done"):
                # Block ALREADY_DONE when the coder was given rejection feedback —
                # you don't get to say "already done" after being told your code
                # is broken.  Force a real retry instead.
                if feedback is not None:
                    console.print(
                        "  [bold yellow]⚠ Coder claimed ALREADY_DONE after "
                        "rejection — nope.  Retrying...[/bold yellow]\n"
                    )
                    continue

                console.print(
                    f"  [cyan]↩ Coder says task is already done. Skipping review.[/cyan]\n"
                )
                review_result = ReviewFeedback(
                    approved=True,
                    summary="Task already satisfied by existing code.",
                ).to_dict()
                break

            # ── O: Handle NEEDS_RESEARCH signal from coder ──
            raw_output = code_result.get("_raw_output", "")
            explanation = code_result.get("explanation", "")
            needs_research_match = re.search(
                r"NEEDS_RESEARCH:\s*(.+)",
                explanation or raw_output or "",
            )
            if needs_research_match and research_requests_this_task < 2:
                query = needs_research_match.group(1).strip()[:200]
                console.print(
                    agent_header(
                        "researcher",
                        f"Coder requested research: {query[:80]}",
                    )
                )
                try:
                    result = await self._timed_await(
                        "researcher",
                        self.agents["researcher"].execute(
                            {
                                "query": query,
                                "task_description": task.get("description", ""),
                                "known_docs_urls": self.known_docs_urls,
                            }
                        ),
                    )
                    confidence = result.get("confidence", "medium")
                    if confidence == "low":
                        log.warning(
                            "Coder-initiated research low confidence for: %s",
                            query[:80],
                        )
                    else:
                        findings = result.get("findings", "")
                        code_ref = result.get("relevant_code", "")
                        key_points = result.get("key_points", [])
                        pip_pkgs = result.get("pip_packages") or {}
                        block = ""
                        if findings:
                            block += findings[:1500]
                        if key_points:
                            block += "\nKey points:\n" + "\n".join(
                                f"  • {p}" for p in key_points[:5]
                            )
                        if code_ref:
                            block += f"\nCode reference:\n{code_ref[:1000]}"
                        if isinstance(pip_pkgs, dict) and pip_pkgs:
                            block += "\npip packages: " + ", ".join(
                                f"{k} → pip install {v}" for k, v in pip_pkgs.items()
                            )
                        if block:
                            self.research_bank[query] = block
                            log.info(
                                "Coder-initiated research cached (%s) for: %s",
                                confidence,
                                query[:80],
                            )
                except Exception as e:
                    log.warning("Coder-initiated research failed: %s", e)

                research_requests_this_task += 1
                # Don't count this as a failure — retry with research in hand
                continue
            elif needs_research_match and research_requests_this_task >= 2:
                log.warning(
                    "Coder hit research request cap (%d) for %s — proceeding",
                    research_requests_this_task,
                    task_id,
                )

            raw_files = code_result.get("files", [])

            # Empty output retry — give the coder one more shot
            if not raw_files or all(
                not f.get("content", "").strip() for f in raw_files
            ):
                if iteration < max_iters - 1:
                    console.print(
                        "  [yellow]⚠ Coder returned empty output. Retrying...[/yellow]\n"
                    )
                    continue
                else:
                    console.print(
                        "  [bold red]✗ Coder returned empty output on final attempt.[/bold red]\n"
                    )
                    break

            # Track the best (non-empty) code we've seen for this task,
            # so ALREADY_DONE on a later global iteration can't nuke it.
            best_code_result = code_result

            # ── Diff-aware: strip unchanged files from coder output ──
            review_code, unchanged_paths = self._strip_unchanged_files(
                code_result,
                existing_results,
            )
            changed_files = review_code.get("files", [])

            log.info("Coder output files: %s", [f["path"] for f in changed_files])
            if code_result.get("explanation"):
                log.info("Coder explanation: %s", code_result["explanation"][:300])

            # Circuit breaker: detect identical consecutive outputs
            # (use raw_files for this — unchanged stripping shouldn't
            # mask genuinely identical coder output)
            full_content = "".join(f.get("content", "") for f in raw_files)
            content_hash = hashlib.md5(full_content.encode()).hexdigest()
            if content_hash == prev_content_hash:
                console.print(
                    "  [bold yellow]⚠ Identical output detected — forcing "
                    "the coder to actually change something.[/bold yellow]"
                )
                failure_history.append(
                    f"Attempt {iteration + 1}: IDENTICAL OUTPUT — coder resubmitted "
                    f"the exact same code as last attempt"
                )
                # Build aggressive feedback that makes the problem impossible to ignore
                feedback = ReviewFeedback(
                    approved=False,
                    summary=(
                        "CRITICAL: Your code is IDENTICAL to your previous submission "
                        "which already failed. You MUST make actual changes. "
                        "Re-read the error messages and fix the specific issues. "
                        "Do NOT just resubmit the same code."
                    ),
                    issues=[
                        ReviewIssue(
                            severity="critical",
                            file="all",
                            description=(
                                "Identical code resubmission detected. The previous "
                                "errors were: " + (last_validation_error or "unknown")
                            ),
                            suggestion="Make concrete changes to address the errors listed above.",
                        )
                    ],
                ).to_dict()
                review_result = feedback
                # Bump temperature to force creativity on next attempt
                coder_agent._temperature_override = min(
                    base_temp + 0.4 * (iteration + 1), 1.2
                )
                continue
            prev_content_hash = content_hash

            # ── AUTO-REPAIR: deterministic fixes before any validation ──
            raw_files_for_repair = code_result.get("files", [])
            _, repairs_made = auto_repair(raw_files_for_repair)
            if repairs_made:
                log.info("Auto-repaired: %s", repairs_made)

            # ── SELF-VERIFY: task-level completeness check ──
            self_verify_result = self_verify(
                code_result.get("files", []),
                task,
            )
            if not self_verify_result.passed:
                console.print(
                    f"  [yellow]⚠ Self-verify: {len(self_verify_result.errors)} issue(s)[/yellow]"
                )
                for err in self_verify_result.errors:
                    log.info("Self-verify: %s — %s", err.file, err.message)

                sv_feedback = ReviewFeedback(
                    approved=False,
                    issues=[
                        ReviewIssue.from_dict(e.to_reviewer_format())
                        for e in self_verify_result.errors
                    ],
                    summary=(
                        f"Self-verification found {len(self_verify_result.errors)} "
                        f"completeness issue(s). Fix these before proceeding."
                    ),
                ).to_dict()

                self.ledger.append(
                    Message(
                        from_agent="self_verifier",
                        to_agent="coder",
                        message_type=MessageType.VALIDATION,
                        content=sv_feedback["summary"],
                        metadata={"task_id": task_id},
                    )
                )

                failure_history.append(
                    f"Attempt {iteration + 1}: SELF-VERIFY — {len(self_verify_result.errors)} completeness issue(s)"
                )
                last_classified = classify_errors(
                    [e.message for e in self_verify_result.errors if e.message]
                )
                feedback = sv_feedback
                review_result = sv_feedback
                continue

            # Log self-verify warnings (non-fatal) for awareness
            for warning in self_verify_result.warnings:
                log.info("Self-verify warning: %s", warning)

            # ── STATIC CHECK: AST-based structural verification ──
            all_files = self._collect_all_files(existing_results, code_result)
            static_result = static_check_files(all_files)

            if not static_result.passed:
                console.print(
                    f"  [yellow]⚠ Static analysis: {len(static_result.errors)} issue(s)[/yellow]"
                )
                for err in static_result.errors:
                    log.info("Static: %s:%s — %s", err.file, err.line, err.message)

                # Build feedback for coder
                static_feedback = ReviewFeedback(
                    approved=False,
                    issues=[
                        ReviewIssue.from_dict(e.to_reviewer_format())
                        for e in static_result.errors
                    ],
                    summary=f"Static analysis found {len(static_result.errors)} structural issue(s) that will cause runtime errors.",
                ).to_dict()

                # Log to ledger
                self.ledger.append(
                    Message(
                        from_agent="static_checker",
                        to_agent="coder",
                        message_type=MessageType.VALIDATION,
                        content=static_feedback["summary"],
                        metadata={"task_id": task_id},
                    )
                )

                failure_history.append(
                    f"Attempt {iteration + 1}: STATIC CHECK — {len(static_result.errors)} issue(s)"
                )

                self.static_check_failures += 1
                # Classify static errors for taxonomy-driven retry
                last_classified = classify_errors(
                    [e.message for e in static_result.errors if e.message]
                )
                feedback = static_feedback
                review_result = static_feedback
                continue

            log.debug("Static analysis passed for %s", task_id)

            # ── SIGNATURE CHECK: verify architect-required signatures exist ──
            required_sigs = task.get("required_signatures", [])
            if required_sigs:
                sig_issues = verify_signatures(all_files, required_sigs)
                if sig_issues:
                    console.print(
                        f"  [yellow]⚠ Missing signatures: {len(sig_issues)} required function(s) not found[/yellow]"
                    )
                    for iss in sig_issues:
                        log.info(
                            "Signature: %s — %s",
                            iss.get("file", "?"),
                            iss.get("description", ""),
                        )

                    sig_feedback = ReviewFeedback(
                        approved=False,
                        issues=[ReviewIssue.from_dict(i) for i in sig_issues],
                        summary=(
                            f"Missing {len(sig_issues)} required function/method signature(s) "
                            f"specified by the architect. These MUST be implemented."
                        ),
                    ).to_dict()

                    self.ledger.append(
                        Message(
                            from_agent="signature_checker",
                            to_agent="coder",
                            message_type=MessageType.VALIDATION,
                            content=sig_feedback["summary"],
                            metadata={"task_id": task_id},
                        )
                    )

                    failure_history.append(
                        f"Attempt {iteration + 1}: MISSING SIGNATURES — {len(sig_issues)} required function(s)"
                    )
                    last_classified = [
                        ClassifiedError(
                            "Missing required signatures",
                            ErrorCategory.SIGNATURE_MISMATCH,
                        )
                    ]
                    feedback = sig_feedback
                    review_result = sig_feedback
                    continue

                log.debug(
                    "All %d required signatures verified for %s",
                    len(required_sigs),
                    task_id,
                )

            # ── FAST VALIDATION: syntax + imports before wasting reviewer time ──
            with self.tracer.span(
                "validate_fast", task_id=task_id, iteration=iteration
            ):
                fast_result = validate_fast(all_files, deps_dir=self.deps_dir)

            if not fast_result.passed:
                # Code is broken — skip the reviewer entirely and feed
                # errors directly back to the coder.
                last_classified = classify_validation_result(fast_result)
                console.print(
                    f"  [bold red]✗ Fast validation FAILED ({len(fast_result.errors)} error(s))[/bold red]"
                )
                for err in fast_result.errors:
                    log.info("Fast validation error: %s", err[:300])

                feedback = fast_result.as_feedback()

                val_parts: list[str] = []
                for e in fast_result.errors:
                    lines = [ln.strip() for ln in e.split("\n") if ln.strip()]
                    val_parts.append(" | ".join(lines[:3])[:200])
                val_summary = "; ".join(val_parts)[:600]
                self.ledger.append(
                    Message(
                        from_agent="validator",
                        to_agent="all",
                        message_type=MessageType.VALIDATION,
                        content=val_summary,
                        metadata={"task_id": task_id},
                    )
                )
                last_validation_error = val_summary
                failure_history.append(
                    f"Attempt {iteration + 1}: SYNTAX/IMPORT — {val_summary[:150]}"
                )
                self.fast_validation_failures += 1

                # Taxonomy-driven research: only research if the error type warrants it
                fast_strategy = get_dominant_strategy(last_classified)
                if fast_strategy.should_research:
                    research_ctx = await self._research_errors(
                        fast_result, all_files, task=task, code_result=code_result
                    )
                    if research_ctx:
                        feedback["research_context"] = research_ctx

                review_result = feedback
                continue

            # ── REVIEWER: code compiles, now check logic/completeness ──
            console.print(
                agent_header("reviewer", f"Reviewing: {task['title']} {iter_label}")
            )
            with self.tracer.span("reviewer", task_id=task_id, iteration=iteration):
                review_result = await self._timed_await(
                    "reviewer",
                    self.agents["reviewer"].execute(
                        {
                            "task": task,
                            "code": review_code,
                            "context": existing_results,
                            "unchanged_files": unchanged_paths,
                            "last_validation_error": last_validation_error,
                            "research_bank": self.research_bank,  # Q: reviewer gets API ref
                        }
                    ),
                )

            approved = review_result.get("approved", False)
            summary = review_result.get("summary", "No summary.")
            issues = review_result.get("issues", [])

            if approved:
                console.print(f"  [green]✓ Approved:[/green] {summary}")

                # ── FULL VALIDATION: run tests now that reviewer is happy ──
                with self.tracer.span(
                    "validate_full", task_id=task_id, iteration=iteration
                ):
                    validation = validate_full(
                        all_files,
                        deps_dir=self.deps_dir,
                        fast_passed=True,
                    )

                if validation.passed:
                    vparts = ["✓ Validation passed"]
                    if validation.tests_ran:
                        vparts.append("tests passed")
                    console.print(f"  [green]{' — '.join(vparts)}[/green]\n")
                    break

                # Tests failed — override reviewer approval
                last_classified = classify_validation_result(validation)
                console.print(
                    f"  [bold red]✗ Validation FAILED ({len(validation.errors)} error(s))[/bold red]"
                )
                for err in validation.errors:
                    log.info("Full validation error: %s", err[:500])

                feedback = validation.as_feedback()

                val_parts_full: list[str] = []
                for e in validation.errors:
                    lines = [ln.strip() for ln in e.split("\n") if ln.strip()]
                    val_parts_full.append(" | ".join(lines[:3])[:200])
                val_summary = "; ".join(val_parts_full)[:600]
                self.ledger.append(
                    Message(
                        from_agent="validator",
                        to_agent="all",
                        message_type=MessageType.VALIDATION,
                        content=val_summary,
                        metadata={"task_id": task_id},
                    )
                )

                self.reviewer_misses += 1
                self.full_validation_failures += 1
                last_validation_error = val_summary

                failure_history.append(
                    f"Attempt {iteration + 1}: VALIDATION — {val_summary[:150]}"
                )

                # Taxonomy-driven research for full validation failures
                full_strategy = get_dominant_strategy(last_classified)
                if full_strategy.should_research:
                    research_ctx = await self._research_errors(
                        validation, all_files, task=task, code_result=code_result
                    )
                    if research_ctx:
                        feedback["research_context"] = research_ctx

                review_result = feedback
                continue

            # Reviewer rejected
            console.print(f"  [red]✗ Rejected:[/red] {summary}")
            for issue in issues:
                log.info(
                    "Reviewer issue [%s]: %s",
                    issue.get("severity", "?"),
                    issue.get("description", ""),
                )

            # Classify reviewer issues so taxonomy drives the next retry
            reviewer_errors = [
                issue.get("description", "")
                for issue in issues
                if issue.get("description")
            ]
            last_classified = (
                classify_errors(reviewer_errors)
                if reviewer_errors
                else [ClassifiedError("Reviewer rejected", ErrorCategory.LOGIC)]
            )

            last_validation_error = None
            rejection_reason = summary[:120]
            if issues:
                rejection_reason = issues[0].get("description", summary)[:120]
            failure_history.append(
                f"Attempt {iteration + 1}: REVIEWER — {rejection_reason}"
            )

            feedback = review_result

        # Reset temperature override so it doesn't leak into the next task
        coder_agent._temperature_override = None

        # If we ended on ALREADY_DONE (empty code) but had real code in a
        # prior iteration, keep the real code so _finalize doesn't lose files.
        final_code = code_result
        if not final_code.get("files") and best_code_result.get("files"):
            final_code = best_code_result

        # Log reviewer accuracy for this task
        if self.reviewer_misses > 0:
            log.info("Reviewer misses so far: %d", self.reviewer_misses)

        return TaskResult(
            task=TaskSpec.from_dict(task),
            code=CodeOutput.from_dict(final_code),
            review=ReviewFeedback.from_dict(review_result),
            iterations=iteration + 1,
            reviewer_misses=self.reviewer_misses,
            static_check_failures=self.static_check_failures,
            fast_validation_failures=self.fast_validation_failures,
            full_validation_failures=self.full_validation_failures,
        ).to_dict()

    # ── Retry hint builder ────────────────────────────────────────────────── #

    @staticmethod
    def _build_retry_hint(failure_history: list[str]) -> str:
        """Scan failure history and build an escalating hint for the coder.

        Returns an empty string if no repeated pattern is detected.
        """
        if len(failure_history) < 2:
            return ""

        hints: list[str] = []

        # ── "Ran 0 tests" repeated ──
        zero_test_count = sum(
            1 for f in failure_history if "Ran 0 tests" in f or "NO TESTS RAN" in f
        )
        if zero_test_count >= 2:
            hints.append(
                "CRITICAL: Tests have failed to run for "
                f"{zero_test_count} consecutive attempts. "
                "The test sandbox may NOT have pytest installed. You MUST "
                "write tests using unittest.TestCase with setUp/tearDown. "
                "Do NOT use @pytest.fixture, tmp_path, or any pytest-specific "
                "features. Use tempfile.mkdtemp() for temp directories and "
                "unittest.mock for mocking."
            )

        # ── Repeated AttributeError on same class ──
        attr_classes: dict[str, int] = {}
        for f in failure_history:
            m = re.search(r"AttributeError.*'(\w+)'", f)
            if m:
                cls = m.group(1)
                attr_classes[cls] = attr_classes.get(cls, 0) + 1
        for cls, count in attr_classes.items():
            if count >= 2:
                hints.append(
                    f"You have used wrong API names for '{cls}' in {count} "
                    f"consecutive attempts. Do NOT guess — only use names "
                    f"from the API REFERENCE section. If no reference is "
                    f"available, request research with NEEDS_RESEARCH or use "
                    f"a simpler approach that avoids this class entirely."
                )

        # ── Repeated STATIC CHECK failures ──
        static_count = sum(1 for f in failure_history if "STATIC CHECK" in f)
        if static_count >= 2:
            hints.append(
                "Static analysis has failed multiple times. Carefully verify "
                "that every variable, class, and function you reference is "
                "either defined in your code or correctly imported."
            )

        # ── Repeated DEPENDENCY ERROR ──
        dep_packages: dict[str, int] = {}
        for f in failure_history:
            dep_m = re.search(r"DEPENDENCY ERROR.*?Package '(\w+)'", f)
            if dep_m:
                pkg = dep_m.group(1)
                dep_packages[pkg] = dep_packages.get(pkg, 0) + 1
        for pkg, count in dep_packages.items():
            if count >= 2:
                hints.append(
                    f"CRITICAL: You have tried to use package '{pkg}' which "
                    f"does not exist on PyPI ({count} attempts). STOP using "
                    f"this package. Either use NEEDS_RESEARCH to find the "
                    f"correct package name, or use a stdlib-only approach."
                )

        if not hints:
            return ""
        return "\n".join(hints)

    # ── Reactive research helpers ───────────────────────────────────────────── #

    def _parse_researchable_errors(
        self,
        errors: list[str],
        all_files: list[dict],
    ) -> list[str]:
        """Extract targeted search queries from validation errors.

        Identifies errors caused by wrong API usage (AttributeError,
        TypeError) and builds queries that include the library name
        so the researcher can look up the correct API.

        Handles both raw tracebacks AND pytest-formatted output like:
          FAILED test_foo.py::test_bar - AttributeError: ...
          E       AttributeError: 'Foo' has no attribute 'bar'
        """
        # Build a set of third-party libraries used in the codebase
        libs: set[str] = set()
        for f in all_files:
            content = f.get("content", "")
            for m in re.finditer(
                r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                content,
                re.M,
            ):
                pkg = m.group(1)
                # Skip stdlib / project-internal modules
                if pkg not in {
                    "os",
                    "sys",
                    "json",
                    "time",
                    "datetime",
                    "pathlib",
                    "typing",
                    "collections",
                    "functools",
                    "itertools",
                    "re",
                    "math",
                    "logging",
                    "abc",
                    "dataclasses",
                    "unittest",
                    "pytest",
                    "tempfile",
                    "shutil",
                    "copy",
                    "__future__",
                }:
                    libs.add(pkg)

        lib_hint = " ".join(sorted(libs)[:3])  # top 3 for query context

        queries: list[str] = []
        seen: set[str] = set()

        for err in errors:
            # NOTE: no 'continue' after any pattern — a single error block
            # (e.g. full pytest output) can contain multiple distinct errors.
            # The `seen` set prevents duplicate queries.

            # ── Pattern 1: AttributeError (first raw match) ──
            m = re.search(
                r"AttributeError:.*'(\w+)'.*has no attribute '(\w+)'",
                err,
            )
            if m:
                cls = m.group(1)
                if cls not in seen:
                    queries.append(
                        f"python {lib_hint} {cls} correct attributes methods API"
                    )
                    seen.add(cls)

            # ── Pattern 2: TypeError with wrong arguments ──
            m = re.search(
                r"TypeError:.*?(\w+\.\w+)\(\).*argument",
                err,
            )
            if m:
                func = m.group(1)
                if func not in seen:
                    queries.append(f"python {lib_hint} {func} method signature API")
                    seen.add(func)

            # ── Pattern 3: ModuleNotFoundError ──
            m = re.search(r"No module named '(\w+)'", err)
            if m:
                mod = m.group(1)
                if mod not in seen:
                    queries.append(f"python {mod} library install usage")
                    seen.add(mod)

            # ── Pattern 4: pytest FAILED line with error type ──
            # FAILED test.py::test_foo - TypeError: ...
            # FAILED test.py::test_foo - NameError: name 'X' is not defined
            for fail_m in re.finditer(
                r"FAILED\s+\S+\s+-\s+(\w+Error):(.+)",
                err,
            ):
                err_type = fail_m.group(1)
                err_msg = fail_m.group(2).strip()
                # Re-parse the error message for known patterns
                attr_m = re.search(
                    r"'(\w+)'.*has no attribute '(\w+)'",
                    err_msg,
                )
                if attr_m:
                    cls = attr_m.group(1)
                    if cls not in seen:
                        queries.append(
                            f"python {lib_hint} {cls} correct attributes methods API"
                        )
                        seen.add(cls)
                elif err_type == "NameError":
                    name_m = re.search(r"name '(\w+)'", err_msg)
                    if name_m and name_m.group(1) not in seen:
                        queries.append(
                            f"python {lib_hint} {name_m.group(1)} correct import usage"
                        )
                        seen.add(name_m.group(1))

            # ── Pattern 5: pytest E-lines (indented error details) ──
            # E       AttributeError: 'ListView' object has no attribute 'set_focus'
            for e_m in re.finditer(
                r"^E\s+(\w+Error):(.+)",
                err,
                re.M,
            ):
                err_type = e_m.group(1)
                err_msg = e_m.group(2).strip()
                attr_m = re.search(
                    r"'(\w+)'.*has no attribute '(\w+)'",
                    err_msg,
                )
                if attr_m:
                    cls = attr_m.group(1)
                    if cls not in seen:
                        queries.append(
                            f"python {lib_hint} {cls} correct attributes methods API"
                        )
                        seen.add(cls)

            # ── Pattern 6: hasattr assertion failures ──
            # AssertionError: assert False
            #  +  where False = hasattr(<MainApp ...>, 'sidebar')
            # This means the test expected an attribute that doesn't exist
            # at that point in the lifecycle — research the class's lifecycle.
            hasattr_m = re.search(
                r"hasattr\(.*?<?(\w+)[ >].*?,\s*'(\w+)'",
                err,
            )
            if hasattr_m:
                cls = hasattr_m.group(1)
                attr = hasattr_m.group(2)
                key = f"lifecycle_{cls}"
                if key not in seen:
                    queries.append(
                        f"python {lib_hint} {cls} lifecycle initialization "
                        f"when is {attr} available"
                    )
                    seen.add(key)

        # ── F: Pattern 7: "Ran 0 tests" / test discovery failure ──
        has_zero_tests = any("Ran 0 tests" in e or "NO TESTS RAN" in e for e in errors)
        if has_zero_tests and "zero_tests" not in seen:
            queries.append(
                "python unittest TestCase test discovery naming "
                "import error conftest __init__"
            )
            seen.add("zero_tests")
            # If we know the framework, also query its test patterns
            if lib_hint:
                queries.append(
                    f"python {lib_hint} testing mock async app unittest TestCase setUp"
                )

        # ── F: Pattern 8: ImportError on test module ──
        for err in errors:
            import_fail = re.search(
                r"(?:ImportError|ModuleNotFoundError):.*Failed to import test module"
                r"|ImportError:.*cannot import name '(\w+)'",
                err,
            )
            if import_fail and "import_test_fail" not in seen:
                name = import_fail.group(1) if import_fail.group(1) else ""
                if name:
                    queries.append(f"python {lib_hint} {name} correct import path")
                else:
                    queries.append(
                        "python test module import error relative import "
                        "conftest __init__ test discovery"
                    )
                seen.add("import_test_fail")

        # ── Fallback: test failures with no specific match ──
        # If tests failed but we couldn't extract specific errors,
        # and we know the main framework, research its test patterns.
        if not queries and lib_hint and any("Test failures" in e for e in errors):
            fallback_lib = sorted(libs)[0] if libs else ""
            if fallback_lib:
                queries.append(
                    f"python {fallback_lib} unittest TestCase testing "
                    f"how to test {fallback_lib} app"
                )

        return queries[:4]  # Cap at 4 queries

    async def _research_errors(
        self,
        validation,  # ValidationResult
        all_files: list[dict],
        task: dict | None = None,
        code_result: dict | None = None,
    ) -> str:
        """Research validation errors and return API reference context.

        Caches results per query so repeated failures for the same error
        don't burn multiple researcher calls.

        Args:
            validation: The ValidationResult with errors.
            all_files: All files in the workspace (for import extraction).
            task: Current task dict (for task_description context).
            code_result: Current coder output (for failing_code extraction).
        """
        queries = self._parse_researchable_errors(validation.errors, all_files)
        if not queries:
            return ""

        # Build error context string from validation errors
        error_context = "\n".join(validation.errors[:5])[:1200]

        # Extract failing code from the files that were just generated
        failing_code = ""
        if code_result:
            for f in code_result.get("files", [])[:3]:
                content = f.get("content", "")
                if content:
                    failing_code += f"# --- {f.get('path', '?')} ---\n{content[:400]}\n"
                if len(failing_code) > 800:
                    break

        task_description = ""
        if task:
            task_description = task.get("description", task.get("title", ""))

        findings_parts: list[str] = []
        for query in queries:
            # ── E: Cache with staleness check ──
            if query in self.research_bank:
                hits = self.research_cache_hits.get(query, 0) + 1
                self.research_cache_hits[query] = hits
                if hits <= 2:
                    log.debug(
                        "Using cached research for: %s (hit %d)", query[:80], hits
                    )
                    findings_parts.append(self.research_bank[query])
                    continue
                else:
                    # Cache is stale — coder is still failing, bust it
                    log.info(
                        "Busting stale research cache for: %s (served %d times)",
                        query[:80],
                        hits,
                    )
                    del self.research_bank[query]
                    del self.research_cache_hits[query]
                    # Modify query to get different results
                    query = query + " API reference documentation site"

            console.print(agent_header("researcher", f"Looking up: {query[:80]}"))
            try:
                result = await self._timed_await(
                    "researcher",
                    self.agents["researcher"].execute(
                        {
                            "query": query,
                            "error_context": error_context,
                            "failing_code": failing_code,
                            "task_description": task_description,
                            "known_docs_urls": self.known_docs_urls,
                        }
                    ),
                )
                confidence = result.get("confidence", "medium")
                findings = result.get("findings", "")
                code_ref = result.get("relevant_code", "")
                key_points = result.get("key_points", [])
                pip_pkgs = result.get("pip_packages") or {}

                # ── D: Only cache medium/high confidence results ──
                if confidence == "low":
                    log.warning(
                        "Low-confidence research for '%s' — not caching",
                        query[:80],
                    )
                    continue

                block = ""
                if findings:
                    block += findings[:1500]
                if key_points:
                    block += "\nKey points:\n" + "\n".join(
                        f"  • {p}" for p in key_points[:5]
                    )
                if code_ref:
                    block += f"\nCode reference:\n{code_ref[:1000]}"
                if isinstance(pip_pkgs, dict) and pip_pkgs:
                    block += "\npip packages: " + ", ".join(
                        f"{k} → pip install {v}" for k, v in pip_pkgs.items()
                    )

                if block:
                    self.research_bank[query] = block
                    findings_parts.append(block)
                    log.info(
                        "Research findings (%s confidence): %s",
                        confidence,
                        findings[:200],
                    )
                else:
                    log.info("No useful research findings for: %s", query)
            except Exception as e:
                log.warning("Research failed for '%s': %s", query, e)

        if not findings_parts:
            return ""

        return (
            "\n\n--- API REFERENCE (from docs research) ---\n"
            + "\n\n".join(findings_parts)
            + "\n--- END API REFERENCE ---\n"
        )

    # ── File collection / validation ──────────────────────────────── #

    @staticmethod
    def _build_workspace_manifest(existing_results: dict) -> str:
        """Build a compact workspace manifest from approved task results.

        Extracts file paths and top-level symbols (classes, functions) using
        AST parsing.  Keeps it compact (<200 tokens) so it doesn't eat too
        much of the coder's context window.

        Returns:
            A string like:
                src/app.py: class App, def main, def setup
                src/models.py: class User, class Post
                tests/test_app.py: class TestApp
        """
        import ast as _ast

        files_by_path: dict[str, str] = {}
        for _task_id, result in existing_results.items():
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content.strip():
                    files_by_path[path] = content

        if not files_by_path:
            return ""

        lines: list[str] = []
        for path in sorted(files_by_path):
            content = files_by_path[path]
            symbols: list[str] = []
            try:
                tree = _ast.parse(content)
                for node in _ast.iter_child_nodes(tree):
                    if isinstance(node, _ast.ClassDef):
                        symbols.append(f"class {node.name}")
                    elif isinstance(node, _ast.FunctionDef | _ast.AsyncFunctionDef):
                        symbols.append(f"def {node.name}")
            except SyntaxError:
                pass  # can't parse, just list the file path

            sym_str = ", ".join(symbols[:8])  # cap symbols per file
            if sym_str:
                lines.append(f"  {path}: {sym_str}")
            else:
                lines.append(f"  {path}")

        # Hard cap to prevent context blowout
        manifest = "\n".join(lines[:30])
        if len(lines) > 30:
            manifest += f"\n  ... and {len(lines) - 30} more files"
        return manifest

    def _collect_all_files(
        self,
        existing_results: dict,
        current_code: dict,
    ) -> list[dict]:
        """Collect all files from existing approved results + current code output.

        Used by the validator to check cross-file dependencies.
        Deduplicates by path, preferring the current code's version.
        """
        files_by_path: dict[str, dict] = {}

        # Add files from ALL previous tasks (not just approved).  When a
        # task fails and gets merged downstream, the coder may not re-output
        # every file.  Without the failed task's files, the validator can't
        # resolve intra-project imports and the deps installer may install
        # unrelated PyPI packages with the same name.  Current code always
        # overrides (latest version wins), so broken code from failed tasks
        # will be superseded if the merged task produces a new version.
        for _task_id, result in existing_results.items():
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                if path and f.get("content", "").strip():
                    files_by_path[path] = f

        # Override with current task's files (latest version wins)
        for f in current_code.get("files", []):
            path = f.get("path", "")
            if path and f.get("content", "").strip():
                files_by_path[path] = f

        return list(files_by_path.values())

    def _validate_outputs(self, results: dict) -> dict:
        """Validate final outputs — last task to write a file wins.

        When multiple tasks produce the same file, we keep the version from
        the task that ran LAST (highest task number), since later tasks build
        on earlier ones. Also runs a basic syntax check for Python files.
        """
        # Build a map: path → (task_id, file_dict) — last writer wins.
        # We iterate in insertion order (task execution order), so the last
        # assignment for each path is the one from the latest task.
        best_files: dict[str, tuple[str, dict]] = {}
        for task_id, result in results.items():
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if not path or not content:
                    continue
                best_files[path] = (task_id, f)  # last writer wins

        # Syntax-check Python files
        for path, (_tid, fdict) in best_files.items():
            if path.endswith(".py"):
                try:
                    compile(fdict["content"], path, "exec")
                    log.debug("Final syntax OK: %s", path)
                except SyntaxError as e:
                    console.print(
                        f"  [bold red]✗ {path} — syntax error: "
                        f"{e.msg} (line {e.lineno})[/bold red]"
                    )

        # Replace results files with the canonical (last-written) versions
        # so _finalize writes the right thing
        merged_result = dict(results)
        for path, (task_id, fdict) in best_files.items():
            if task_id in merged_result:
                code = merged_result[task_id].get("code", {})
                code["files"] = [
                    fdict if f.get("path") == path else f for f in code.get("files", [])
                ]

        return merged_result

    def _generate_requirements(self, results: dict) -> None:
        """Generate requirements.txt from all third-party imports across approved tasks.

        Purely for the user's convenience — the coder never sees this file.
        Lists the packages that were auto-installed during the run.
        """
        from the_bois.tools.deps import extract_imports

        all_files: list[dict] = []
        for result in results.values():
            if not result.get("review", {}).get("approved", False):
                continue
            for f in result.get("code", {}).get("files", []):
                if f.get("path", "").endswith(".py") and f.get("content", "").strip():
                    all_files.append(f)

        if not all_files:
            return

        imports = extract_imports(all_files)
        if not imports:
            return

        # Check what's actually installed in deps_dir
        deps_path = Path(self.deps_dir)
        installed: set[str] = set()
        if deps_path.is_dir():
            # Read the installed packages from the deps directory
            for item in deps_path.iterdir():
                name = item.name
                if name.endswith(".dist-info"):
                    # Extract package name from dist-info directory
                    pkg_name = name.rsplit("-", 1)[0] if "-" in name else name
                    installed.add(pkg_name.replace("-", "_").lower())

        # Write requirements.txt
        req_lines = sorted(imports)
        if req_lines:
            req_path = self.workspace.path / "requirements.txt"
            req_path.write_text("\n".join(req_lines) + "\n")
            console.print(
                f"  [dim]→ workspace/requirements.txt ({len(req_lines)} packages)[/dim]"
            )
            log.info("Generated requirements.txt: %s", req_lines)

    async def _finalize(self, results: dict) -> None:
        """Save all code files to workspace and persist the ledger."""
        console.print(Rule("[bold]Output[/bold]"))

        # Deduplicate: last writer wins (matches _validate_outputs strategy).
        # Since results is ordered by task execution, the last assignment
        # for each path is the canonical version.
        final_files: dict[str, str] = {}
        for task_id, result in results.items():
            code = result.get("code", {})
            for f in code.get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content:
                    final_files[path] = content  # last writer wins

        saved: list[str] = []
        for path, content in final_files.items():
            self.workspace.write_file(path, content)
            saved.append(path)

        if saved:
            console.print("[bold]Files saved to workspace:[/bold]")
            for s in saved:
                console.print(f"  [dim]→ workspace/{s}[/dim]")
        else:
            console.print("[dim]No files to save.[/dim]")

        console.print()

    async def _learn(
        self,
        scope: str,
        tasks: list[dict],
        all_results: dict,
    ) -> None:
        """Post-run learning — score the run, extract lessons, persist to memory."""
        if not self.memory or not self.memory.enabled:
            return

        try:
            console.print(Rule("[bold]🧠 Learning[/bold]"))
            metrics = await self.memory.learn_from_run(
                client=self.client,
                run_id=self.run_id,
                scope=scope,
                tasks=tasks,
                all_results=all_results,
            )

            # Print learning summary
            rate = metrics.get("approval_rate", 0)
            passed = metrics.get("tasks_passed", 0)
            total = metrics.get("tasks_total", 0)
            console.print(
                f"  [dim]Run score:[/dim] {passed}/{total} tasks approved "
                f"({rate:.0%} approval rate)"
            )

            stats = self.memory.stats
            console.print(
                f"  [dim]Memory:[/dim] {stats['examples']} examples, "
                f"{stats['mistakes']} mistake patterns, "
                f"{stats['episodes']} episodes stored"
            )
            console.print()
        except Exception as e:
            console.print(f"  [yellow]⚠ Learning failed (non-fatal): {e}[/yellow]")

    def _checkpoint(self, all_results: dict) -> None:
        """Save incremental checkpoint — approved files + metadata to workspace.

        Called after each approved task so overnight runs survive crashes.
        The checkpoint can be resumed with ``--from workspace/run_xxx/``.
        """
        # Write approved files
        approved_count = 0
        file_count = 0
        for _tid, result in all_results.items():
            if not result.get("review", {}).get("approved", False):
                continue
            approved_count += 1
            for f in result.get("code", {}).get("files", []):
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content:
                    self.workspace.write_file(path, content)
                    file_count += 1

        # Save structured checkpoint metadata (task status, review summaries)
        checkpoint_data = {}
        for tid, result in all_results.items():
            if tid.startswith("_"):
                continue
            checkpoint_data[tid] = {
                "task": result.get("task", {}),
                "approved": result.get("review", {}).get("approved", False),
                "review_summary": result.get("review", {}).get("summary", ""),
                "iterations": result.get("iterations", 0),
            }
        ckpt_path = self.workspace.path / "checkpoint.json"
        ckpt_path.write_text(json.dumps(checkpoint_data, indent=2))

        # Ledger snapshot
        self.ledger.save(self.workspace.path / "ledger.json")

        log.info(
            "Checkpoint: %d task(s) approved, %d file(s) saved",
            approved_count,
            file_count,
        )

    def _save_state(self) -> None:
        """Persist the ledger for debugging / resume."""
        ledger_path = self.workspace.path / "ledger.json"
        self.ledger.save(ledger_path)
        log.info("Ledger saved: %s (%d messages)", ledger_path, self.ledger.count)
