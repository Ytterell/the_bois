"""Main orchestration loop — coordinates agents to solve a given scope."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

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
from the_bois.memory import MemoryStore
from the_bois.memory.ledger import Ledger, Message, MessageType
from the_bois.models.ollama import OllamaClient
from the_bois.tools.repair import auto_repair
from the_bois.tools.validator import validate_code, validate_fast, validate_full
from the_bois.tools.workspace import Workspace

console = Console()

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

    def __init__(self, config: Config, client=None) -> None:
        self.config = config
        self.ledger = Ledger()
        self.client = client or OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
            keep_alive=config.ollama.keep_alive,
        )
        # Each run gets its own timestamped subfolder
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = Path(config.workspace.path) / f"run_{self.run_id}"
        self.workspace = Workspace(run_path)
        self.agents: dict = {}
        self.research_bank: dict[str, str] = {}  # query → findings (persists entire run)
        self.reviewer_misses: int = 0  # times reviewer approved but validation rejected

        # Memory system
        self.memory: MemoryStore | None = None
        if config.memory.enabled:
            self.memory = MemoryStore(config.memory)

    def _init_agents(self) -> None:
        """Instantiate all agents with their configs."""
        args = lambda name: (
            name, self.config.agents[name], self.client, self.ledger, self.memory,
        )

        self.agents = {
            "coordinator": CoordinatorAgent(*args("coordinator")),
            "architect": ArchitectAgent(*args("architect")),
            "coder": CoderAgent(*args("coder")),
            "reviewer": ReviewerAgent(*args("reviewer")),
            "researcher": ResearcherAgent(*args("researcher")),
        }

    async def run(self, scope: str, seed_files: dict[str, str] | None = None) -> None:
        """Main entry — takes a problem scope and runs the agent loop.

        Args:
            scope: The problem description.
            seed_files: Optional dict of {path: content} from a previous run
                        to resume/iterate on.
        """
        if not await self.client.health_check():
            console.print("[bold red]✗ Cannot connect to Ollama. Is it running?[/bold red]")
            return

        self._init_agents()

        try:
            await self._execute_pipeline(scope, seed_files=seed_files)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted — saving current state...[/yellow]")
        finally:
            self._save_state()
            await self.client.close()

    async def _execute_pipeline(
        self, scope: str, seed_files: dict[str, str] | None = None,
    ) -> None:
        """The full Coordinator → Researcher → Architect → Coder ↔ Reviewer → Coordinator pipeline."""

        # ── Build seed context from previous run (if any) ──
        seed_context = ""
        seed_results: dict = {}
        if seed_files:
            seed_file_list = [
                {"path": p, "content": c} for p, c in seed_files.items()
            ]
            seed_results["_seed"] = {
                "task": {"id": "_seed", "title": "Previous run output", "description": ""},
                "code": {"files": seed_file_list},
                "review": {"approved": True, "issues": [], "summary": "Seed from previous run."},
                "iterations": 0,
            }
            seed_context = "\n\nEXISTING CODE FROM A PREVIOUS ATTEMPT (focus on what " \
                "needs fixing, completing, or improving — do NOT redo what already works):\n"
            for p, c in seed_files.items():
                seed_context += f"\n--- {p} ---\n{c}\n"

        # ── Step 0a: Coordinator analyzes scope ──
        console.print(agent_header("coordinator", "Analyzing scope for clarity..."))
        scope_with_seed = scope + seed_context if seed_context else scope
        scope_analysis = await self.agents["coordinator"].analyze_scope(scope_with_seed)
        refined_scope = scope_analysis.get("refined_scope", scope)
        needs_research = scope_analysis.get("needs_research", False)
        research_queries = scope_analysis.get("research_queries", [])

        console.print(f"  [dim]Refined scope:[/dim] {refined_scope[:200]}")
        if needs_research:
            console.print(f"  [magenta]Research needed: {len(research_queries)} query(ies)[/magenta]")
        else:
            console.print("  [dim]No research needed — proceeding directly.[/dim]")
        console.print()

        # ── Step 0b: Researcher gathers context (if needed) ──
        research_findings: list[dict] = []
        if needs_research and research_queries:
            for query in research_queries:
                console.print(agent_header("researcher", f"Researching: {query}"))
                try:
                    result = await self.agents["researcher"].execute({"query": query})
                    research_findings.append({"query": query, **result})
                    findings_preview = result.get("findings", "")[:150]
                    console.print(f"  [dim]{findings_preview}[/dim]")

                    # Store in persistent research bank for coder access
                    block = ""
                    findings = result.get("findings", "")
                    code_ref = result.get("relevant_code", "")
                    key_points = result.get("key_points", [])
                    if findings:
                        block += findings[:800]
                    if key_points:
                        block += "\nKey points:\n" + "\n".join(
                            f"  • {p}" for p in key_points[:5]
                        )
                    if code_ref:
                        block += f"\nCode reference:\n{code_ref[:500]}"
                    if block:
                        self.research_bank[query] = block

                except Exception as e:
                    console.print(f"  [yellow]⚠ Research failed: {e}[/yellow]")

            if self.research_bank:
                console.print(
                    f"  [magenta]📚 Research bank: {len(self.research_bank)} "
                    f"topic(s) stored for coder reference[/magenta]"
                )
            console.print()

        # ── Step 1: Architect decomposes the problem ──
        console.print(agent_header("architect", "Analyzing scope and creating task plan..."))
        architect_scope = refined_scope + seed_context if seed_context else refined_scope
        plan = await self.agents["architect"].execute(
            {"scope": architect_scope, "research": research_findings}
        )
        tasks = plan.get("tasks", [])

        if not tasks:
            console.print("[bold red]Architect produced no tasks. Aborting.[/bold red]")
            return

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

            console.print(Rule(f"[bold]Global Iteration {global_iter + 1}/{max_global}[/bold]"))

            failed_deps = False
            for task in tasks:
                # Check if this task's dependencies were all approved
                deps = task.get("dependencies", [])
                unmet = [
                    d for d in deps
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
                    console.print(
                        f"  [yellow]⚠ Merging failed dep(s) {unmet} into {task['id']}[/yellow]"
                    )

                # Build context: only relevant files for this task
                filtered_context = self._build_coder_context(all_results, current_task=task)

                result = await self._execute_task(task, filtered_context)
                all_results[task["id"]] = result

                # Auto-checkpoint after approved tasks — overnight insurance
                if result.get("review", {}).get("approved", False):
                    self._checkpoint(all_results)
                else:
                    console.print(
                        f"  [yellow]⚠ Task {task['id']} failed all review attempts.[/yellow]"
                    )

            # ── Step 3: Coordinator reviews everything ──
            console.print(agent_header("coordinator", "Reviewing overall progress..."))
            decision = await self.agents["coordinator"].execute(
                {"scope": refined_scope, "plan": plan, "results": all_results}
            )

            dec = decision.get("decision", "approve")
            reason = decision.get("reason", "No reason given.")
            console.print(f"  [dim]Decision:[/dim] [bold]{dec}[/bold] — {reason}\n")

            if dec == "approve":
                console.print("[bold green]✅ The bois are satisfied. Work complete.[/bold green]\n")
                break

            elif dec == "rework":
                rework_ids = set(decision.get("tasks_to_rework", []))
                tasks = [t for t in plan.get("tasks", []) if t["id"] in rework_ids]
                if not tasks:
                    console.print("[yellow]Coordinator requested rework but listed no valid task IDs. Approving.[/yellow]")
                    break
                console.print(f"[yellow]↻ Reworking {len(tasks)} task(s): {', '.join(t['id'] for t in tasks)}[/yellow]\n")

            elif dec == "replan":
                console.print("[yellow]↻ Coordinator wants a full replan...[/yellow]\n")
                console.print(agent_header("architect", "Re-planning based on feedback..."))
                plan = await self.agents["architect"].execute(
                    {"scope": refined_scope, "feedback": reason, "research": research_findings}
                )
                tasks = plan.get("tasks", [])
                if not tasks:
                    console.print("[bold red]Architect produced no tasks on replan. Stopping.[/bold red]")
                    break
                # Preserve approved results across replans — don't nuke
                # work that already passed review
                all_results = {
                    tid: res for tid, res in all_results.items()
                    if res.get("review", {}).get("approved", False)
                }

        else:
            console.print(
                f"[bold yellow]⚠ Hit max global iterations ({max_global}). "
                f"Outputting best effort.[/bold yellow]\n"
            )

        # ── Step 4: Validate before saving ──
        all_results = self._validate_outputs(all_results)

        # ── Step 5: Save results ──
        await self._finalize(all_results)

        # ── Step 6: Learn from this run ──
        await self._learn(refined_scope, tasks, all_results)

    def _build_coder_context(
        self, all_results: dict, current_task: dict | None = None,
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
            path_mentions = set(re.findall(r'[\w./]+\.(?:py|js|ts|json|yaml|yml|toml|cfg|txt|md)', desc))
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
                if (
                    path in dep_paths
                    or any(m in path for m in mentioned_paths)
                ):
                    filtered[path] = fdict

            # If filtering leaves nothing (no matches), fall back to all
            if filtered:
                files_by_path = filtered

        # Single entry with all unique files — coder/reviewer see each
        # file exactly once regardless of how many tasks touched it.
        if files_by_path:
            context["_codebase"] = {
                "code": {"files": list(files_by_path.values())},
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
        self, code_result: dict, existing_results: dict,
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
            console.print(
                f"  [dim]📎 Stripped {len(unchanged_paths)} unchanged file(s): "
                f"{', '.join(unchanged_paths)}[/dim]"
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

        for iteration in range(max_iters):
            iter_label = f"(attempt {iteration + 1}/{max_iters})"

            # Temperature ramp: nudge the coder to try different approaches
            # on retries instead of repeating the same mistake.
            if iteration > 0:
                ramped = min(base_temp + 0.2 * iteration, 1.0)
                coder_agent._temperature_override = ramped
                console.print(f"  [dim]🌡 Temperature ramped to {ramped:.1f}[/dim]")
            else:
                coder_agent._temperature_override = None

            # Coder writes
            console.print(agent_header("coder", f"Task: {task['title']} {iter_label}"))
            code_result = await coder_agent.execute(
                {
                    "task": task,
                    "feedback": feedback,
                    "context": existing_results,
                    "failure_history": failure_history,
                    "research_bank": self.research_bank,
                }
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
                review_result = {
                    "approved": True,
                    "issues": [],
                    "summary": "Task already satisfied by existing code.",
                }
                break

            raw_files = code_result.get("files", [])

            # Empty output retry — give the coder one more shot
            if not raw_files or all(not f.get("content", "").strip() for f in raw_files):
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
                code_result, existing_results,
            )
            changed_files = review_code.get("files", [])

            for f in changed_files:
                console.print(f"  [green]📄 {f['path']}[/green]")
            if code_result.get("explanation"):
                console.print(f"  [dim]{code_result['explanation'][:200]}[/dim]")

            # Circuit breaker: detect identical consecutive outputs
            # (use raw_files for this — unchanged stripping shouldn't
            # mask genuinely identical coder output)
            full_content = "".join(f.get("content", "") for f in raw_files)
            content_hash = hashlib.md5(full_content.encode()).hexdigest()
            if content_hash == prev_content_hash:
                console.print(
                    "  [bold yellow]⚠ Circuit breaker: coder produced identical "
                    "output. Aborting task.[/bold yellow]"
                )
                break
            prev_content_hash = content_hash

            # ── AUTO-REPAIR: deterministic fixes before any validation ──
            raw_files_for_repair = code_result.get("files", [])
            _, repairs_made = auto_repair(raw_files_for_repair)
            if repairs_made:
                console.print(f"  [cyan]🔧 Auto-repaired: {', '.join(repairs_made)}[/cyan]")

            # ── FAST VALIDATION: syntax + imports before wasting reviewer time ──
            all_files = self._collect_all_files(existing_results, code_result)
            console.print("  [dim]⚙ Fast validation (syntax + imports)...[/dim]")
            fast_result = validate_fast(all_files)

            if not fast_result.passed:
                # Code is broken — skip the reviewer entirely and feed
                # errors directly back to the coder.
                console.print("  [bold red]✗ Fast validation FAILED — skipping reviewer[/bold red]")
                for err in fast_result.errors:
                    for line in err.split("\n")[:3]:
                        console.print(f"    [red]{line[:150]}[/red]")
                console.print()

                feedback = fast_result.as_feedback()

                val_parts: list[str] = []
                for e in fast_result.errors:
                    lines = [ln.strip() for ln in e.split("\n") if ln.strip()]
                    val_parts.append(" | ".join(lines[:3])[:200])
                val_summary = "; ".join(val_parts)[:600]
                self.ledger.append(Message(
                    from_agent="validator",
                    to_agent="all",
                    message_type=MessageType.VALIDATION,
                    content=val_summary,
                    metadata={"task_id": task_id},
                ))
                last_validation_error = val_summary
                failure_history.append(
                    f"Attempt {iteration + 1}: SYNTAX/IMPORT — {val_summary[:150]}"
                )

                research_ctx = await self._research_errors(fast_result, all_files)
                if research_ctx:
                    feedback["research_context"] = research_ctx

                review_result = feedback
                continue

            console.print("  [green]✓ Syntax + imports OK[/green]")

            # ── REVIEWER: code compiles, now check logic/completeness ──
            console.print(agent_header("reviewer", f"Reviewing: {task['title']} {iter_label}"))
            review_result = await self.agents["reviewer"].execute(
                {
                    "task": task,
                    "code": review_code,
                    "context": existing_results,
                    "unchanged_files": unchanged_paths,
                    "last_validation_error": last_validation_error,
                }
            )

            approved = review_result.get("approved", False)
            summary = review_result.get("summary", "No summary.")
            issues = review_result.get("issues", [])

            if approved:
                console.print(f"  [green]✓ Approved:[/green] {summary}")

                # ── FULL VALIDATION: run tests now that reviewer is happy ──
                console.print("  [dim]⚙ Full validation (+ tests)...[/dim]")
                validation = validate_full(all_files)

                if validation.passed:
                    vparts = ["✓ Validation passed"]
                    if validation.tests_ran:
                        vparts.append("tests passed")
                    console.print(f"  [green]{' — '.join(vparts)}[/green]\n")
                    break

                # Tests failed — override reviewer approval
                console.print("  [bold red]✗ Full validation FAILED (tests crash/fail)[/bold red]")
                for err in validation.errors:
                    for line in err.split("\n")[:3]:
                        console.print(f"    [red]{line[:150]}[/red]")
                console.print()

                feedback = validation.as_feedback()

                val_parts_full: list[str] = []
                for e in validation.errors:
                    lines = [ln.strip() for ln in e.split("\n") if ln.strip()]
                    val_parts_full.append(" | ".join(lines[:3])[:200])
                val_summary = "; ".join(val_parts_full)[:600]
                self.ledger.append(Message(
                    from_agent="validator",
                    to_agent="all",
                    message_type=MessageType.VALIDATION,
                    content=val_summary,
                    metadata={"task_id": task_id},
                ))

                self.reviewer_misses += 1
                last_validation_error = val_summary

                failure_history.append(
                    f"Attempt {iteration + 1}: VALIDATION — {val_summary[:150]}"
                )

                research_ctx = await self._research_errors(validation, all_files)
                if research_ctx:
                    feedback["research_context"] = research_ctx

                review_result = feedback
                continue

            # Reviewer rejected
            console.print(f"  [red]✗ Rejected:[/red] {summary}")
            for issue in issues:
                sev = issue.get("severity", "?")
                desc = issue.get("description", "")
                console.print(f"    [red]• [{sev}][/red] {desc}")
            console.print()

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
            console.print(
                f"  [dim yellow]⚠ Reviewer missed {self.reviewer_misses} "
                f"runtime error(s) so far this run[/dim yellow]"
            )

        return {
            "task": task,
            "code": final_code,
            "review": review_result,
            "iterations": iteration + 1,
            "reviewer_misses": self.reviewer_misses,
        }

    # ── Reactive research helpers ─────────────────────────────────── #

    def _parse_researchable_errors(
        self, errors: list[str], all_files: list[dict],
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
                r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", content, re.M,
            ):
                pkg = m.group(1)
                # Skip stdlib / project-internal modules
                if pkg not in {
                    "os", "sys", "json", "time", "datetime", "pathlib",
                    "typing", "collections", "functools", "itertools",
                    "re", "math", "logging", "abc", "dataclasses",
                    "unittest", "pytest", "tempfile", "shutil", "copy",
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
                r"AttributeError:.*'(\w+)'.*has no attribute '(\w+)'", err,
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
                r"TypeError:.*?(\w+\.\w+)\(\).*argument", err,
            )
            if m:
                func = m.group(1)
                if func not in seen:
                    queries.append(
                        f"python {lib_hint} {func} method signature API"
                    )
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
                r"FAILED\s+\S+\s+-\s+(\w+Error):(.+)", err,
            ):
                err_type = fail_m.group(1)
                err_msg = fail_m.group(2).strip()
                # Re-parse the error message for known patterns
                attr_m = re.search(
                    r"'(\w+)'.*has no attribute '(\w+)'", err_msg,
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
                r"^E\s+(\w+Error):(.+)", err, re.M,
            ):
                err_type = e_m.group(1)
                err_msg = e_m.group(2).strip()
                attr_m = re.search(
                    r"'(\w+)'.*has no attribute '(\w+)'", err_msg,
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
                r"hasattr\(.*?<?(\w+)[ >].*?,\s*'(\w+)'", err,
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

        # ── Fallback: test failures with no specific match ──
        # If tests failed but we couldn't extract specific errors,
        # and we know the main framework, research its test patterns.
        if not queries and lib_hint and any("Test failures" in e for e in errors):
            fallback_lib = sorted(libs)[0] if libs else ""
            if fallback_lib:
                queries.append(
                    f"python {fallback_lib} testing best practices "
                    f"how to test {fallback_lib} app correctly"
                )

        return queries[:3]  # Cap at 3 queries

    async def _research_errors(
        self,
        validation,  # ValidationResult
        all_files: list[dict],
    ) -> str:
        """Research validation errors and return API reference context.

        Caches results per query so repeated failures for the same error
        don't burn multiple researcher calls.
        """
        queries = self._parse_researchable_errors(validation.errors, all_files)
        if not queries:
            return ""

        findings_parts: list[str] = []
        for query in queries:
            # Check cache first
            if query in self.research_bank:
                console.print(f"  [dim]🔎 Using cached research: {query[:60]}[/dim]")
                findings_parts.append(self.research_bank[query])
                continue

            console.print(agent_header("researcher", f"Looking up: {query[:80]}"))
            try:
                result = await self.agents["researcher"].execute({"query": query})
                findings = result.get("findings", "")
                code_ref = result.get("relevant_code", "")
                key_points = result.get("key_points", [])

                block = ""
                if findings:
                    block += findings[:800]
                if key_points:
                    block += "\nKey points:\n" + "\n".join(
                        f"  • {p}" for p in key_points[:5]
                    )
                if code_ref:
                    block += f"\nCode reference:\n{code_ref[:500]}"

                if block:
                    self.research_bank[query] = block
                    findings_parts.append(block)
                    console.print(f"  [dim]{findings[:120]}[/dim]")
                else:
                    console.print("  [yellow]No useful findings.[/yellow]")
            except Exception as e:
                console.print(f"  [yellow]⚠ Research failed: {e}[/yellow]")

        if not findings_parts:
            return ""

        return (
            "\n\n--- API REFERENCE (from docs research) ---\n"
            + "\n\n".join(findings_parts)
            + "\n--- END API REFERENCE ---\n"
        )

    # ── File collection / validation ──────────────────────────────── #

    def _collect_all_files(
        self, existing_results: dict, current_code: dict,
    ) -> list[dict]:
        """Collect all files from existing approved results + current code output.

        Used by the validator to check cross-file dependencies.
        Deduplicates by path, preferring the current code's version.
        """
        files_by_path: dict[str, dict] = {}

        # Add files from previously approved tasks
        for _task_id, result in existing_results.items():
            if result.get("review", {}).get("approved", False):
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
                    console.print(f"  [green]✓ {path} — syntax OK[/green]")
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
                    fdict if f.get("path") == path else f
                    for f in code.get("files", [])
                ]

        return merged_result

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
        self, scope: str, tasks: list[dict], all_results: dict,
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

        console.print(
            f"  [dim]💾 Checkpoint: {approved_count} task(s) approved, "
            f"{file_count} file(s) saved[/dim]"
        )

    def _save_state(self) -> None:
        """Persist the ledger for debugging / resume."""
        ledger_path = self.workspace.path / "ledger.json"
        self.ledger.save(ledger_path)
        console.print(f"[dim]Ledger saved: {ledger_path} ({self.ledger.count} messages)[/dim]")
