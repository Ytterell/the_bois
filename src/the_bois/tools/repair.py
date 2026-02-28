"""Auto-repair pipeline — deterministic fixes that don't need an LLM round-trip.

Some validation failures are purely mechanical: Unicode chars, missing
__init__.py, trailing whitespace.  Fixing these programmatically saves
a full coder iteration and reduces the chance of the model "fixing" one
thing while breaking another.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from the_bois.tools.validator import sanitize_code


def auto_repair(files: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply deterministic fixes to generated code files.

    Args:
        files: List of {"path": str, "content": str} dicts.

    Returns:
        (fixed_files, list_of_repairs_made)
        The original list is mutated in-place for efficiency.
    """
    repairs: list[str] = []

    # Track which packages need __init__.py
    packages_seen: set[str] = set()

    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        if not content:
            continue

        original = content

        # ── 1. Unicode → ASCII (delegates to sanitize_code) ──
        content = sanitize_code(content)

        # ── 2. Strip trailing whitespace from all lines ──
        lines = content.split("\n")
        stripped = [line.rstrip() for line in lines]
        if stripped != lines:
            content = "\n".join(stripped)

        # ── 3. Collapse 3+ consecutive blank lines → 2 ──
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        # ── 4. Ensure file ends with exactly one newline ──
        content = content.rstrip("\n") + "\n"

        # ── 5. Fix common f-string escaping: unmatched braces ──
        # Models sometimes write f"value is {x" (missing closing brace).
        # This is a best-effort fix for trivial cases only.
        # Count unmatched { in f-strings — too risky to auto-fix generally,
        # so we just log it as a warning.

        # Track package directories for __init__.py generation
        if path.endswith(".py"):
            parts = PurePosixPath(path).parts[:-1]  # directory segments
            for i in range(len(parts)):
                pkg = "/".join(parts[: i + 1])
                packages_seen.add(pkg)

        if content != original:
            f["content"] = content
            repairs.append(f"Cleaned {path} (whitespace/unicode/formatting)")

    # ── 6. Auto-generate missing __init__.py for packages ──
    existing_paths = {f.get("path", "") for f in files}
    for pkg in sorted(packages_seen):
        init_path = f"{pkg}/__init__.py"
        if init_path not in existing_paths:
            files.append({"path": init_path, "content": ""})
            repairs.append(f"Generated missing {init_path}")
            existing_paths.add(init_path)

    return files, repairs
