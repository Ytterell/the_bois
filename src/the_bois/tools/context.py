"""Context optimization utilities — smarter context management for agents.

This module provides utilities for:
- File chunking: extracting relevant sections from large files
- Context profiles: defining what each agent needs in context
- Smart windowing: managing conversation history within token budgets
- Progressive compression: squeezing code context to fit token budgets
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from the_bois.utils import estimate_tokens

log = logging.getLogger(__name__)


# Default max file size before chunking (in lines)
DEFAULT_MAX_FILE_LINES = 500


@dataclass
class ChunkedFile:
    """A file that has been chunked for context."""

    path: str
    content: str
    is_chunked: bool = False
    original_lines: int = 0
    kept_lines: int = 0


def extract_relevant_code(
    content: str,
    file_path: str,
    task_description: str,
    max_lines: int = DEFAULT_MAX_FILE_LINES,
) -> ChunkedFile:
    """Extract relevant code sections from a large file based on task.

    Uses AST to find function/class definitions and filters based on:
    1. Functions/classes explicitly mentioned in task
    2. Functions/classes used by mentioned items

    If file is small enough, returns it unchanged.
    """
    lines = content.split("\n")
    line_count = len(lines)

    # If file is small enough, return as-is
    if line_count <= max_lines:
        return ChunkedFile(
            path=file_path,
            content=content,
            is_chunked=False,
            original_lines=line_count,
            kept_lines=line_count,
        )

    # Parse and find relevant definitions
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        # Can't parse, return truncated content
        return ChunkedFile(
            path=file_path,
            content=content,
            is_chunked=True,
            original_lines=line_count,
            kept_lines=max_lines,
        )

    # Extract keywords from task description
    keywords = _extract_keywords(task_description)

    # Find all definitions (functions, classes)
    definitions: list[
        tuple[int, int, str, str]
    ] = []  # (start_line, end_line, type, name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                definitions.append(
                    (
                        node.lineno,
                        node.end_lineno or node.lineno,
                        type(node).__name__,
                        node.name,
                    )
                )

    # Score each definition by relevance
    scored: list[tuple[int, int, str, int]] = []  # (start, end, name, score)
    for start, end, def_type, name in definitions:
        score = 0
        name_lower = name.lower()

        # Direct mention in keywords
        if name_lower in keywords:
            score += 100

        # Partial match
        for kw in keywords:
            if kw in name_lower:
                score += 10

        # Classes and their methods get a boost
        if def_type == "ClassDef":
            score += 5

        scored.append((start, end, name, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[3], reverse=True)

    # Take top definitions until we fit in max_lines
    selected_ranges: list[tuple[int, int]] = []
    selected_lines = 0

    for start, end, name, score in scored:
        if score == 0:
            continue
        range_lines = end - start + 1
        if selected_lines + range_lines <= max_lines or not selected_ranges:
            selected_ranges.append((start, end))
            selected_lines += range_lines

    if not selected_ranges:
        # No relevant code found, take first max_lines
        return ChunkedFile(
            path=file_path,
            content="\n".join(lines[:max_lines]),
            is_chunked=True,
            original_lines=line_count,
            kept_lines=max_lines,
        )

    # Build chunked content
    # Convert 1-indexed lines to 0-indexed
    selected_ranges = sorted(selected_ranges)
    kept_lines_list: list[str] = []

    # Add any imports at the top (lines before first function/class)
    first_def_start = selected_ranges[0][0]
    if first_def_start > 1:
        # Include imports/docstrings at top
        import_end = min(10, first_def_start - 1)
        kept_lines_list.extend(lines[:import_end])
        if import_end < first_def_start - 1:
            kept_lines_list.append("    # ... (imports truncated) ...")

    # Add selected definitions
    for start, end in selected_ranges:
        kept_lines_list.extend(lines[start - 1 : end])

    chunked_content = "\n".join(kept_lines_list)

    return ChunkedFile(
        path=file_path,
        content=chunked_content,
        is_chunked=True,
        original_lines=line_count,
        kept_lines=len(kept_lines_list),
    )


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from task description."""
    # Extract identifiers (function names, class names, variables)
    identifiers = set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", text.lower()))

    # Remove common stopwords
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "function",
        "class",
        "method",
        "return",
        "param",
        "args",
        "kwargs",
        "import",
        "from",
        "define",
        "implement",
        "create",
        "add",
        "make",
        "use",
        "using",
        "file",
        "files",
        "module",
        "package",
        "data",
        "test",
        "tests",
        "write",
        "read",
        "save",
        "load",
        "handle",
    }
    identifiers -= stopwords

    return identifiers


def chunk_files_for_context(
    files: list[dict],
    task_description: str,
    max_file_lines: int = DEFAULT_MAX_FILE_LINES,
) -> list[dict]:
    """Chunk large files to fit within context limits.

    Args:
        files: List of {"path": str, "content": str} dicts
        task_description: Description of current task (for relevance filtering)
        max_file_lines: Maximum lines per file before chunking

    Returns:
        List of files with large files chunked
    """
    result: list[dict] = []

    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")

        if not content:
            result.append(f)
            continue

        # Only chunk Python files
        if not path.endswith(".py"):
            result.append(f)
            continue

        chunked = extract_relevant_code(content, path, task_description, max_file_lines)

        if chunked.is_chunked:
            # Track that this was chunked for logging
            result.append(
                {
                    "path": path,
                    "content": chunked.content,
                    "_chunked": True,
                    "_original_lines": chunked.original_lines,
                    "_kept_lines": chunked.kept_lines,
                }
            )
        else:
            result.append(f)

    return result


# ── Progressive code compression ─────────────────────────────────────
#
# When the coder's prompt is too fat to fit in the context window, we
# progressively compress the *existing codebase* section (the biggest
# compressible chunk).  Four levels:
#
#   0 — full code (unchanged)
#   1 — strip docstrings, comments, consecutive blank lines (~30% smaller)
#   2 — signatures only: keep def/class lines + decorators, body → `...`
#   3 — names only: bare `def foo(…):` and `class Bar:` one-liners

_COMPRESSION_LABELS = {
    0: "full",
    1: "stripped (no comments/docstrings)",
    2: "signatures only",
    3: "names only",
}


def _strip_comments_and_docstrings(source: str) -> str:
    """Level 1: remove comments, docstrings, and collapse blank lines.

    Uses AST to safely identify string-expression-statements (docstrings),
    falls back to regex for non-Python or unparseable files.
    """
    lines = source.split("\n")

    # Try AST-based docstring removal for Python
    try:
        tree = ast.parse(source)
        # Collect line ranges of docstrings (Expr nodes whose value is a Constant str)
        docstring_lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and hasattr(node, "lineno")
                and hasattr(node, "end_lineno")
            ):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    docstring_lines.add(ln)

        cleaned: list[str] = []
        prev_blank = False
        for i, line in enumerate(lines, 1):
            # Skip docstring lines
            if i in docstring_lines:
                continue
            # Strip inline comments (but not inside strings — good enough heuristic)
            stripped = line.rstrip()
            if "#" in stripped:
                # Only strip if # is not inside a string literal
                code_part = stripped.split("#", 1)[0].rstrip()
                # Heuristic: if the quote count before # is even, it's a real comment
                if code_part.count('"') % 2 == 0 and code_part.count("'") % 2 == 0:
                    stripped = code_part
            # Collapse consecutive blank lines
            is_blank = not stripped.strip()
            if is_blank and prev_blank:
                continue
            prev_blank = is_blank
            cleaned.append(stripped)

        return "\n".join(cleaned)
    except SyntaxError:
        # Non-Python or broken syntax — just strip # comments and blank runs
        cleaned = []
        prev_blank = False
        for line in lines:
            s = line.rstrip()
            is_blank = not s.strip()
            if is_blank and prev_blank:
                continue
            prev_blank = is_blank
            cleaned.append(s)
        return "\n".join(cleaned)


def _signatures_only(source: str) -> str:
    """Level 2: keep class/function signatures, replace bodies with `...`.

    Preserves imports, top-level assignments, decorators, and class
    hierarchy.  Gives the coder enough to know what exists and how
    to call it, without the implementation noise.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse — fall back to level-1 compression
        return _strip_comments_and_docstrings(source)

    lines = source.split("\n")
    kept: list[str] = []

    # Always keep imports and top-level simple statements (first ~30 lines or
    # until the first class/function def)
    first_def_line = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno"):
                first_def_line = node.lineno
                break

    header_end = min(first_def_line or len(lines), len(lines))
    # Grab imports / top-level assigns (up to first def, max 40 lines)
    for i, line in enumerate(lines[: min(header_end - 1, 40)], 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            kept.append(line)

    # Walk top-level definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Grab decorator lines + def line
            start = (
                node.decorator_list[0].lineno
                if node.decorator_list and hasattr(node.decorator_list[0], "lineno")
                else node.lineno
            )
            # The def line itself (may span multiple lines for long signatures)
            def_end = node.lineno
            for ln in range(start - 1, min(def_end + 2, len(lines))):
                line = lines[ln]
                kept.append(line)
                if line.rstrip().endswith(":"):
                    break
            kept.append("    ...")
            kept.append("")

        elif isinstance(node, ast.ClassDef):
            # Class header
            start = (
                node.decorator_list[0].lineno
                if node.decorator_list and hasattr(node.decorator_list[0], "lineno")
                else node.lineno
            )
            for ln in range(start - 1, min(node.lineno + 1, len(lines))):
                kept.append(lines[ln])

            # Methods inside class — signatures only
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c_start = (
                        child.decorator_list[0].lineno
                        if child.decorator_list
                        and hasattr(child.decorator_list[0], "lineno")
                        else child.lineno
                    )
                    for ln in range(c_start - 1, min(child.lineno + 2, len(lines))):
                        line = lines[ln]
                        kept.append(line)
                        if line.rstrip().endswith(":"):
                            break
                    kept.append("        ...")
            kept.append("")

    return "\n".join(kept)


def _names_only(source: str) -> str:
    """Level 3: bare-minimum — just def/class names.

    For when context is *extremely* tight.  Gives the coder a table of
    contents so it doesn't accidentally shadow existing names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Extract def/class lines with regex as last resort
        result: list[str] = []
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                result.append(
                    stripped.split("(")[0] + "(...)" if "(" in stripped else stripped
                )
        return "\n".join(result) if result else "(unparseable file)"

    kept: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            base_str = f"({bases})" if bases else ""
            kept.append(f"class {node.name}{base_str}: ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            # Just params, no annotations to save space
            params = ", ".join(arg.arg for arg in node.args.args)
            kept.append(f"{prefix} {node.name}({params}): ...")

    return "\n".join(kept) if kept else "(empty module)"


def compress_code_context(
    files: list[dict],
    max_tokens: int,
) -> tuple[list[dict], int]:
    """Progressively compress code files to fit within a token budget.

    Tries compression levels 0 → 3, stopping at the first level where
    the total token count fits.  Non-Python files are kept as-is (they're
    usually small config/data files).

    Args:
        files: List of {"path": str, "content": str} dicts.
        max_tokens: Target token budget for all files combined.

    Returns:
        (compressed_files, level_used) where level_used is 0-3.
    """
    from the_bois.utils import estimate_tokens as _estimate

    compressors = [
        None,  # level 0: no-op
        _strip_comments_and_docstrings,  # level 1
        _signatures_only,  # level 2
        _names_only,  # level 3
    ]

    compressed: list[dict] = []
    total_tokens = 0

    for level, compressor in enumerate(compressors):
        compressed = []
        total_tokens = 0

        for f in files:
            path = f.get("path", "")
            content = f.get("content", "")

            if compressor and path.endswith(".py") and content:
                content = compressor(content)

            compressed.append({"path": path, "content": content})
            total_tokens += _estimate(content)

        if total_tokens <= max_tokens:
            if level > 0:
                log.info(
                    "Compressed code context to level %d (%s): %d tokens",
                    level,
                    _COMPRESSION_LABELS[level],
                    total_tokens,
                )
            return compressed, level

    # Even level 3 doesn't fit — return it anyway, it's the best we can do
    log.warning(
        "Code context (%d tokens) exceeds budget (%d) even at max compression",
        total_tokens,
        max_tokens,
    )
    return compressed, 3


# Context profiles: what each agent type needs in context
AGENT_CONTEXT_PROFILES = {
    "coordinator": {
        "needs_scope": True,
        "needs_plan": True,
        "needs_results": True,
        "needs_history": True,
        "max_history_messages": 15,
    },
    "architect": {
        "needs_scope": True,
        "needs_research": True,
        "needs_history": False,
        "max_history_messages": 5,
    },
    "coder": {
        "needs_codebase": True,
        "needs_task": True,
        "needs_history": True,
        "max_history_messages": 10,
        "needs_feedback": True,
    },
    "reviewer": {
        "needs_codebase": True,
        "needs_task": True,
        "needs_history": True,
        "max_history_messages": 10,
        "needs_previous_results": True,
    },
    "researcher": {
        "needs_scope": True,
        "needs_history": False,
    },
}


def get_context_for_agent(agent_name: str) -> dict:
    """Get context requirements for a specific agent."""
    return AGENT_CONTEXT_PROFILES.get(agent_name, {})
