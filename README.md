# the_bois

Local multi-agent LLM system powered by Ollama / Ollama-Vulkan. Five
specialized agents collaborate in a pipeline to solve programming tasks —
no cloud APIs, no telemetry, just your hardware and a handful of models
that argue with each other until the code works.

## How It Works

A problem scope goes in. The Coordinator analyzes it, the Researcher digs
up docs if needed, the Architect breaks it into tasks, the Coder writes
the implementation, the Reviewer tears it apart, and the Validator
actually runs it. If something fails, the loop retries with escalating
temperature until the agents produce something that compiles, passes
tests, and survives review — or until the iteration budget runs out.

The system learns across runs: gold examples, anti-patterns, and full
episode histories are persisted to a JSON-backed memory layer and injected
into future prompts via embedding similarity search.

## Architecture

```
                         +---------------------+
                         |    CLI (typer/rich)  |
                         +----------+----------+
                                    |
                                    v
                         +----------+----------+
                         |    Orchestrator      |
                         |  (pipeline driver)   |
                         +----------+----------+
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
    +---------+---------+                       +---------+---------+
    |    Coordinator     |                       |     Researcher    |
    | (scope analysis,   |                       | (PyPI, GitHub,    |
    |  convergence)      |                       |  DuckDuckGo)      |
    +---------+---------+                       +---------+---------+
              |                                           |
              v                                           |
    +---------+---------+                                 |
    |     Architect      |<--- research findings ---------+
    | (task decompose,   |
    |  2-6 subtasks)     |
    +---------+---------+
              |
              |  for each task:
              v
    +---------+---------+     +-------------------+
    |       Coder        |---->|  Auto-Repair      |
    | (plan + implement, |     | (unicode, ws,     |
    |  streaming output) |     |  __init__.py)     |
    +---------+---------+     +--------+----------+
              |                         |
              v                         v
    +---------+-------------------------+---------+
    |              Validator (sandbox)             |
    |  validate_fast(): syntax + imports          |
    |  validate_full(): + pytest execution        |
    |  (RLIMIT_AS, RLIMIT_CPU, RLIMIT_FSIZE)     |
    +---------+-----------------------------------+
              |
              | if syntax OK:
              v
    +---------+---------+
    |      Reviewer      |
    | (logic, complete-  |
    |  ness, style)      |
    +---------+---------+
              |
              | approved + tests pass --> checkpoint
              | rejected / tests fail --> retry with feedback
              v
    +---------+---------+
    |    Coordinator     |
    |  approve / rework  |
    |  / replan          |
    +-------------------+
```

### Pipeline Flow

```
Coordinator.analyze_scope()
  |
  +--> Researcher.execute()            [if external libs detected]
  |
  +--> Architect.execute()             [decomposes into 2-6 tasks]
  |
  +--> for each task:
  |      |
  |      +--> Coder.execute()          [two-phase: plan then code]
  |      |      |
  |      |      +--> auto_repair()     [deterministic fixes]
  |      |      +--> validate_fast()   [syntax + imports]
  |      |      |
  |      |      +--> Reviewer.execute()
  |      |      |      |
  |      |      |      +--> validate_full()  [+ pytest]
  |      |      |
  |      |      +--> (retry if rejected, up to max_task_iterations)
  |      |
  |      +--> checkpoint on approval
  |
  +--> Coordinator.execute()           [approve / rework / replan]
         |
         +--> approve  --> done
         +--> rework   --> retry specific tasks
         +--> replan   --> Architect re-decomposes
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (or Ollama-Vulkan for GPU compute) running locally
- Enough VRAM to run the models in your `config.yaml` (default config
  uses qwen2.5 14b/32b variants — budget ~20GB+ VRAM for comfortable
  operation)

Pull the models before your first run:

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5-coder:32b-instruct-q5_K_M
ollama pull qwen2.5-coder:14b
ollama pull nomic-embed-text
```

## Installation

```bash
git clone <repo-url> && cd the_bois
pip install -e .
```

This installs the `the-bois` CLI as an entry point.

## Usage

All commands assume Ollama is running (`ollama serve`).

### Run a task

```bash
the-bois run "Build a CLI tool that converts CSV to JSON"
```

### Resume from a previous run

Each run checkpoints to `workspace/run_<timestamp>/`. Resume with:

```bash
the-bois run "same or updated scope" --from workspace/run_2026-03-01_00-32-28/
```

### Dry run (no Ollama needed)

Exercises the full pipeline with mock LLM responses. Useful for testing
changes to the orchestration, validation, or agent plumbing:

```bash
the-bois run "anything" --dry-run
```

### Smoke test (real models, trivial task)

A quick sanity check that models are loaded and the pipeline works
end-to-end (~2-3 minutes):

```bash
the-bois run "anything" --smoke
```

### Seed memory

Pre-load the memory system with gold/anti examples from a JSON file:

```bash
the-bois seed-memory memory/seed_examples.json
```

Each entry needs: `agent`, `role` ("gold"/"anti"), `task_description`,
`output_snippet`, and optionally `rejection_reason`.

### Check Ollama status

```bash
the-bois status
```

## Configuration

`config.yaml` in the project root. Structure:

```yaml
agents:
  coordinator:
    model: "qwen2.5:14b"
    temperature: 0.2
    num_ctx: 16384        # context window size
    num_predict: 4096     # max output tokens
  coder:
    model: "qwen2.5-coder:32b-instruct-q5_K_M"
    temperature: 0.15
    timeout: 2400         # per-agent timeout override (seconds)
  # ... (architect, reviewer, researcher follow the same schema)

orchestration:
  max_task_iterations: 5    # retries per task before giving up
  max_global_iterations: 7  # full pipeline loops
  context_max_messages: 20  # ledger messages per agent context
  global_timeout: 14400     # 4 hour wall-clock limit

ollama:
  base_url: "http://localhost:11434"
  timeout: 600
  keep_alive: "15m"

memory:
  enabled: true
  path: "./memory"
  embedding_model: "nomic-embed-text"
  max_examples: 200
  max_injection_tokens: 500
```

If `config.yaml` is missing, defaults from `config.py:DEFAULTS` are used
(smaller models: mistral:7b, phi3:3.8b — runs on weaker hardware but
produces worse output).

## Memory System

Three JSON-file-backed layers, all using Ollama embeddings
(`nomic-embed-text`) for similarity-based retrieval:

**ExampleBank** — Gold (first-try approval) and anti (total failure)
examples per agent. Retrieved by embedding similarity to the current
task description. Capped at `max_examples`.

**MistakeJournal** — Frequency-tracked anti-patterns per agent. Uses
cosine similarity (threshold > 0.85) for fuzzy dedup. Only surfaces
warnings when a pattern has been seen 2+ times.

**EpisodeStore** — Full run histories indexed by scope embedding. Gives
the coordinator and architect "last time we tried something like this"
context.

Memory injection happens in `memory/injection.py`, called by
`BaseAgent.think()`. Memory context is prepended to the user prompt,
capped at 35% of the remaining context window budget.

## Validation

All generated code is executed in a sandboxed subprocess with resource
limits:

- `RLIMIT_AS`: 512 MB address space
- `RLIMIT_CPU`: 30 seconds CPU time
- `RLIMIT_FSIZE`: 50 MB max file writes
- `RLIMIT_NPROC`: 50 max child processes

Two validation tiers:

1. **validate_fast()** — Syntax check + import check. Runs before the
   reviewer sees the code. If this fails, the reviewer is skipped
   entirely and errors go straight back to the coder.

2. **validate_full()** — Everything in fast, plus pytest execution. Runs
   after reviewer approval. If tests fail, the reviewer's approval is
   overridden and the coder gets another shot.

An auto-repair step (`tools/repair.py`) runs before validation:
Unicode-to-ASCII replacement, trailing whitespace cleanup, blank line
collapse, missing `__init__.py` generation. This prevents burning a
coder iteration on purely mechanical issues.

## Project Structure

```
the_bois/
  config.yaml                  # per-agent models, temperatures, limits
  pyproject.toml               # package metadata, dependencies, entry point
  memory/                      # persistent memory (JSON files)
    examples.json              # gold/anti examples
    mistakes.json              # anti-pattern frequency tracker
    episodes/                  # full run histories
    episode_index.json         # scope embeddings for episode retrieval
    seed_examples.json         # manual seed data
  workspace/                   # run outputs (gitignored)
    run_<timestamp>/           # each run gets its own folder
      *.py                     # generated code files
      checkpoint.json          # task status snapshot
      ledger.json              # full message log
  src/the_bois/
    cli.py                     # typer CLI (run, seed-memory, status)
    config.py                  # dataclass configs, YAML loader, defaults
    orchestrator.py            # main pipeline loop, retry logic
    utils.py                   # JSON parsing, token estimation, file parsing
    agents/
      base.py                  # BaseAgent ABC (think, think_stream, ledger)
      coordinator.py           # scope analysis, approve/rework/replan
      architect.py             # task decomposition
      coder.py                 # two-phase code generation (plan + implement)
      researcher.py            # PyPI / GitHub / DuckDuckGo search
      reviewer.py              # code review with structured issue output
    memory/
      embeddings.py            # Ollama embedding calls
      examples.py              # ExampleBank (gold/anti with retrieval)
      mistakes.py              # MistakeJournal (frequency-tracked patterns)
      episodes.py              # EpisodeStore (run history RAG)
      injection.py             # memory context builder for agent prompts
      ledger.py                # append-only message log, task-scoped filtering
      scoring.py               # run scoring, lesson extraction
    models/
      ollama.py                # OllamaClient (chat, chat_stream, embeddings)
      mock.py                  # MockOllamaClient for dry-run / testing
    tools/
      repair.py                # deterministic auto-repair (unicode, whitespace)
      search.py                # web search utilities
      validator.py             # sandbox execution, syntax/import/test checks
      workspace.py             # file I/O for run output directories
```

## Key Design Decisions

**Delimiter-based code output** — The coder uses `---FILE: path---` /
`---END---` delimiters instead of JSON. Local models are unreliable JSON
producers at scale; delimiters are more robust for multi-file output.
Parsed by `utils.parse_delimited_files()`.

**Two-phase coder** — On first attempt, `think()` generates a plan, then
`think_stream()` generates code with streaming progress. On retries
(feedback present), planning is skipped to avoid wasting tokens
re-explaining the same approach.

**Temperature ramping** — The orchestrator bumps coder temperature by
+0.2 per retry iteration (capped at 1.0) to encourage different
approaches. Reset between tasks.

**Prompt ordering** — Local models attend most strongly to content that
appears last. The coder prompt places reference material first, then
context, then failure history, then the task description. Reordering
breaks output quality.

**Diff-aware reviews** — `_strip_unchanged_files()` hashes file contents
and only sends changed files to the reviewer. Prevents token waste on
code the reviewer has already approved.

**Reactive research** — When validation fails with `AttributeError`,
`TypeError`, or `ModuleNotFoundError`, the orchestrator extracts
targeted queries from the traceback and dispatches the researcher
mid-task to fetch correct API docs. Results are cached in
`research_bank` for the duration of the run.

**Safe fallbacks** — Agent responses that fail to parse JSON default to
rejection/rework, never silent approval. The system would rather retry
than ship broken code it can't verify.

## Dependencies

Runtime: `httpx`, `ddgs` (DuckDuckGo search), `typer`, `pyyaml`,
`rich`, `pytest`. All declared in `pyproject.toml`.

No external database. Everything is JSON files + Ollama / Ollama-Vulkan.

## License

TBD
