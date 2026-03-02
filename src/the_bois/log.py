"""Logging setup — two-tier output so the terminal stays readable.

File handler (DEBUG) captures everything: Ollama metrics, token budgets,
validation details, context decisions.  Console handler only fires for
WARNING+ from Python logging callers (httpx retries, tight budgets, etc.)
— the rich console.print() calls in the orchestrator handle the clean
progress view.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging(run_dir: Path | str) -> Path:
    """Configure root logger with file + console handlers.

    Call once at startup before the orchestrator runs.

    Args:
        run_dir: Workspace run directory (e.g. workspace/run_2026-03-02_21-48-17).
                 The log file lands at ``<run_dir>/run.log``.

    Returns:
        Path to the log file.
    """
    global _configured
    if _configured:
        return Path(run_dir) / "run.log"

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── File handler: everything, for post-mortem debugging ──
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(fh)

    # ── Console handler: warnings+ only (retries, budget issues) ──
    # The rich console.print() calls handle the pretty progress view;
    # this is just for stdlib logging callers that hit WARNING or above.
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True
    return log_path
