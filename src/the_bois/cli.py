"""CLI entry point — the front door to the bois."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="the-bois",
    help="Local multi-agent LLM system — the bois got your back.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    scope: str = typer.Argument(..., help="The problem or scope for the bois to solve"),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to config file"
    ),
    from_run: Path | None = typer.Option(
        None, "--from", help="Path to a previous run folder to build upon"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Mock LLM calls — exercises the full pipeline in seconds",
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="Real models, trivial scope, minimal iterations (~2-3 min)",
    ),
) -> None:
    """Submit a problem scope for the bois to tackle."""
    from the_bois.config import load_config
    from the_bois.orchestrator import Orchestrator

    console.print(
        Panel(
            f"[bold cyan]{scope}[/bold cyan]",
            title="[bold]🤖 the_bois — assembling the crew[/bold]",
            border_style="cyan",
        )
    )

    # Load seed files from a previous run if --from is provided
    seed_files: dict[str, str] = {}
    if from_run:
        if not from_run.is_dir():
            console.print(f"[bold red]✗ Not a directory: {from_run}[/bold red]")
            raise typer.Exit(1)

        # Skip metadata files when loading seed code
        _SKIP_FILES = {"ledger.json", "checkpoint.json"}

        for fpath in from_run.rglob("*"):
            if fpath.is_file() and fpath.name not in _SKIP_FILES:
                rel = str(fpath.relative_to(from_run))
                try:
                    seed_files[rel] = fpath.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue  # skip binary / unreadable files

        # Show checkpoint info if available
        ckpt_path = from_run / "checkpoint.json"
        if ckpt_path.exists():
            import json
            try:
                ckpt = json.loads(ckpt_path.read_text())
                approved = sum(1 for v in ckpt.values() if v.get("approved"))
                console.print(
                    f"[bold yellow]↻ Resuming from checkpoint: "
                    f"{approved}/{len(ckpt)} task(s) were approved[/bold yellow]"
                )
            except (json.JSONDecodeError, OSError):
                pass

        console.print(
            f"[bold yellow]↻ Resuming from {from_run} "
            f"({len(seed_files)} file(s) loaded)[/bold yellow]\n"
        )

    config = load_config(config_path)

    # ── Dry-run mode: mock LLM, full pipeline, zero Ollama dependency ──
    client = None
    if dry_run:
        from the_bois.models.mock import MockOllamaClient

        client = MockOllamaClient()
        console.print(
            "[bold magenta]🧪 DRY-RUN MODE — using MockOllamaClient "
            "(no Ollama needed)[/bold magenta]\n"
        )

    # ── Smoke mode: real models, trivial scope, fewer iterations ──
    if smoke:
        scope = (
            "Implement a Python function `add(a: int, b: int) -> int` in `adder.py` "
            "that returns the sum of its arguments. Write pytest tests in `test_adder.py`."
        )
        config.orchestration.max_task_iterations = 2
        config.orchestration.max_global_iterations = 1
        console.print(
            "[bold magenta]🔥 SMOKE MODE — trivial scope, "
            f"max {config.orchestration.max_task_iterations} task iters, "
            f"{config.orchestration.max_global_iterations} global iter[/bold magenta]\n"
        )

    orchestrator = Orchestrator(config, client=client)
    asyncio.run(orchestrator.run(scope, seed_files=seed_files or None))


@app.command(name="seed-memory")
def seed_memory(
    examples_file: Path = typer.Argument(
        ..., help="JSON file with gold/anti examples to seed into memory",
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to config file",
    ),
) -> None:
    """Seed the memory system with gold examples from a JSON file.

    Each entry needs: agent, role ("gold"/"anti"), task_description, output_snippet.
    Embeddings are computed automatically.
    """
    import json

    from the_bois.config import load_config
    from the_bois.memory import MemoryStore
    from the_bois.models.ollama import OllamaClient

    if not examples_file.exists():
        console.print(f"[bold red]\u2717 File not found: {examples_file}[/bold red]")
        raise typer.Exit(1)

    try:
        entries = json.loads(examples_file.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[bold red]\u2717 Invalid JSON: {e}[/bold red]")
        raise typer.Exit(1)

    if not isinstance(entries, list):
        console.print("[bold red]\u2717 Expected a JSON array of examples.[/bold red]")
        raise typer.Exit(1)

    config = load_config(config_path)
    if not config.memory.enabled:
        console.print("[bold red]\u2717 Memory is disabled in config.[/bold red]")
        raise typer.Exit(1)

    async def _seed() -> None:
        client = OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
        )
        if not await client.health_check():
            console.print("[bold red]\u2717 Cannot connect to Ollama.[/bold red]")
            await client.close()
            return

        memory = MemoryStore(config.memory)
        added = 0
        for entry in entries:
            agent = entry.get("agent", "coder")
            role = entry.get("role", "gold")
            task_desc = entry.get("task_description", "")
            snippet = entry.get("output_snippet", "")
            rejection = entry.get("rejection_reason", "")

            if not task_desc:
                console.print(f"  [yellow]\u26a0 Skipping entry with no task_description[/yellow]")
                continue

            console.print(f"  [dim]Embedding: {task_desc[:80]}...[/dim]")
            await memory.examples.add_example(
                client=client,
                agent=agent,
                role=role,
                task_description=task_desc,
                output_snippet=snippet,
                rejection_reason=rejection,
            )
            added += 1

        await client.close()
        console.print(
            f"\n[bold green]\u2713 Seeded {added} example(s) into memory "
            f"({memory.examples.count} total)[/bold green]"
        )

    asyncio.run(_seed())


@app.command()
def status() -> None:
    """Check Ollama connection and list available models."""
    from the_bois.models.ollama import OllamaClient

    async def _check() -> None:
        client = OllamaClient()
        healthy = await client.health_check()
        if not healthy:
            console.print("[bold red]✗ Ollama is not running![/bold red]")
            console.print("  Start it with: [bold]ollama serve[/bold]")
            await client.close()
            return

        console.print("[bold green]✓ Ollama is running[/bold green]")
        models = await client.list_models()
        console.print(f"\n[bold]Available models ({len(models)}):[/bold]")
        for model in models:
            console.print(f"  • {model}")
        await client.close()

    asyncio.run(_check())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
