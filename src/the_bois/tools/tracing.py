"""Run tracer — lightweight span-based tracing to JSON Lines.

Records timing spans for pipeline stages, agent calls, and validation
steps.  Writes to ``trace.jsonl`` in the run workspace directory.

Each line is a self-contained JSON object:
    {"span_id": "...", "name": "coder", "start": 1710000000.123,
     "end": 1710000005.456, "duration_ms": 5333, "task_id": "task_1", ...}

Use ``the-bois trace <run_dir>`` to view a human-readable summary.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RunTracer:
    """Append-only span tracer that writes JSON Lines to disk.

    Usage::

        tracer = RunTracer(workspace_path / "trace.jsonl")

        with tracer.span("coder", task_id="task_1") as s:
            result = await coder.execute(...)
            s.set("tokens_generated", 1234)

    Spans can be nested — the tracer tracks a parent stack automatically.
    """

    def __init__(self, trace_path: Path) -> None:
        self._path = trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._parent_stack: list[str] = []
        self._spans: list[dict] = []  # in-memory copy for summary

    @contextmanager
    def span(self, name: str, **metadata: Any):
        """Context manager that records a timed span.

        Args:
            name: Span name (e.g. "coder", "validate_fast", "coordinator").
            **metadata: Arbitrary key-value pairs attached to the span.

        Yields a ``SpanHandle`` that allows adding metadata mid-span.
        """
        span_id = uuid.uuid4().hex[:12]
        parent_id = self._parent_stack[-1] if self._parent_stack else None
        handle = SpanHandle(metadata)

        self._parent_stack.append(span_id)
        start = time.perf_counter()
        wall_start = time.time()

        try:
            yield handle
        finally:
            elapsed = time.perf_counter() - start
            self._parent_stack.pop()

            record = {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start": round(wall_start, 3),
                "end": round(wall_start + elapsed, 3),
                "duration_ms": round(elapsed * 1000),
                **metadata,
                **handle.extra,
            }

            # Mark failures if the span exited with an exception
            # (contextmanager re-raises, so we check handle)
            if handle.error:
                record["error"] = str(handle.error)

            self._spans.append(record)
            self._write_line(record)

    def _write_line(self, record: dict) -> None:
        """Append a single JSON line to the trace file."""
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            log.warning("Failed to write trace span to %s", self._path)

    def summary(self) -> dict[str, Any]:
        """Build an in-memory summary of all recorded spans.

        Returns a dict with total duration, per-name aggregates, and
        the full span list.
        """
        if not self._spans:
            return {"total_ms": 0, "by_name": {}, "spans": []}

        by_name: dict[str, dict[str, Any]] = {}
        for s in self._spans:
            name = s["name"]
            dur = s.get("duration_ms", 0)
            if name not in by_name:
                by_name[name] = {"count": 0, "total_ms": 0, "max_ms": 0}
            entry = by_name[name]
            entry["count"] += 1
            entry["total_ms"] += dur
            entry["max_ms"] = max(entry["max_ms"], dur)

        # Total = duration of the longest top-level span, or sum of root spans
        root_spans = [s for s in self._spans if s.get("parent_id") is None]
        total_ms = sum(s.get("duration_ms", 0) for s in root_spans)

        return {
            "total_ms": total_ms,
            "by_name": by_name,
            "span_count": len(self._spans),
        }

    @staticmethod
    def load_trace(trace_path: Path) -> list[dict]:
        """Load spans from a trace JSONL file."""
        spans: list[dict] = []
        if not trace_path.exists():
            return spans
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return spans

    @staticmethod
    def render_summary(spans: list[dict]) -> str:
        """Render a human-readable summary from loaded spans.

        Returns a multi-line string suitable for console output.
        """
        if not spans:
            return "No trace spans found."

        by_name: dict[str, dict[str, Any]] = {}
        for s in spans:
            name = s["name"]
            dur = s.get("duration_ms", 0)
            if name not in by_name:
                by_name[name] = {"count": 0, "total_ms": 0, "max_ms": 0, "errors": 0}
            entry = by_name[name]
            entry["count"] += 1
            entry["total_ms"] += dur
            entry["max_ms"] = max(entry["max_ms"], dur)
            if s.get("error"):
                entry["errors"] += 1

        root_spans = [s for s in spans if s.get("parent_id") is None]
        total_ms = sum(s.get("duration_ms", 0) for s in root_spans)

        lines: list[str] = []
        lines.append(f"Trace: {len(spans)} spans, total {_fmt_ms(total_ms)}")
        lines.append("")
        lines.append(
            f"{'Stage':<25} {'Count':>6} {'Total':>10} {'Max':>10} {'Errors':>7}"
        )
        lines.append("-" * 62)

        for name, stats in sorted(by_name.items(), key=lambda x: -x[1]["total_ms"]):
            lines.append(
                f"{name:<25} {stats['count']:>6} "
                f"{_fmt_ms(stats['total_ms']):>10} "
                f"{_fmt_ms(stats['max_ms']):>10} "
                f"{stats['errors']:>7}"
            )

        # Show task-level breakdown
        task_spans = [s for s in spans if s.get("task_id")]
        if task_spans:
            lines.append("")
            lines.append("Per-task breakdown:")
            tasks: dict[str, int] = {}
            for s in task_spans:
                tid = s["task_id"]
                tasks[tid] = tasks.get(tid, 0) + s.get("duration_ms", 0)
            for tid, ms in sorted(tasks.items(), key=lambda x: -x[1]):
                lines.append(f"  {tid}: {_fmt_ms(ms)}")

        return "\n".join(lines)


class SpanHandle:
    """Mutable handle yielded by ``RunTracer.span()``."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self.extra: dict[str, Any] = dict(initial) if initial else {}
        self.error: BaseException | None = None

    def set(self, key: str, value: Any) -> None:
        """Add or update a metadata field on the current span."""
        self.extra[key] = value


def _fmt_ms(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    mins, secs_r = divmod(int(secs), 60)
    return f"{mins}m{secs_r}s"
