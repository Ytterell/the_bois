"""Auto-dependency installer — because 'just pip install it' doesn't work in a sandbox.

Extracts third-party imports from generated code, resolves the correct
PyPI package name via the API (no hardcoded dicts, we're not animals),
and installs them into a sandboxed deps directory.

The deps_dir persists per run so packages installed for task_1 are
available for task_2 without re-downloading.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from the_bois.tools.search import resolve_pypi_name

log = logging.getLogger(__name__)

# Python 3.10+ has sys.stdlib_module_names.  For older versions we'd
# need a fallback, but the project requires 3.11+ so we're good.
_STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)


# ── Import extraction ────────────────────────────────────────────────

def extract_imports(files: list[dict]) -> set[str]:
    """Extract third-party top-level import names from generated Python files.

    Uses `ast` (not regex, we're civilised) to find `import X` and
    `from X import Y` statements.  Filters out:
      - stdlib modules (sys, os, json, etc.)
      - intra-project modules (files that exist in the generated set)

    Returns a set of top-level package names like {"textual", "flask"}.
    """
    # Build set of "project-internal" module names from the file set
    internal_modules: set[str] = set()
    for f in files:
        path = f.get("path", "")
        if path.endswith(".py"):
            # foo/bar/baz.py → foo (top-level package)
            parts = Path(path).parts
            if parts:
                internal_modules.add(parts[0])
            # Also the stem: baz.py → baz
            internal_modules.add(Path(path).stem)

    third_party: set[str] = set()

    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        if not path.endswith(".py") or not content.strip():
            continue

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            continue  # broken code — validator will catch it

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in _STDLIB_MODULES and top not in internal_modules:
                        third_party.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in _STDLIB_MODULES and top not in internal_modules:
                        third_party.add(top)

    return third_party


# ── Package name resolution ──────────────────────────────────────────

async def resolve_package_name(import_name: str) -> str | None:
    """Resolve an import name to its pip-installable package name.

    Thin wrapper around the PyPI lookup in search.py.  Returns the
    canonical package name or None if the package doesn't exist.
    """
    return await resolve_pypi_name(import_name)


# ── Dependency installer ─────────────────────────────────────────────

@dataclass
class DepsResult:
    """What happened when we tried to install dependencies."""
    installed: list[str] = field(default_factory=list)
    already_available: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, error)


def _is_importable(module_name: str, extra_paths: list[str] | None = None) -> bool:
    """Check if a module is importable without actually importing it in-process.

    Runs a quick subprocess so we don't pollute the orchestrator's namespace.
    """
    env_paths = ":".join(extra_paths) if extra_paths else ""
    check_script = (
        f"import sys; sys.path[:0] = {extra_paths or []}; "
        f"import importlib.util; "
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True, timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


async def ensure_deps(
    imports: set[str],
    deps_dir: str,
    venv_sp: str | None = None,
) -> DepsResult:
    """Install missing third-party packages into deps_dir.

    For each import:
      1. Check if it's already importable (from venv or deps_dir) → skip
      2. Resolve the PyPI package name via the API
      3. pip install --target {deps_dir}
      4. Report installed / failed

    Returns a DepsResult so the caller knows what happened.
    """
    result = DepsResult()
    if not imports:
        return result

    extra_paths = [deps_dir]
    if venv_sp:
        extra_paths.append(venv_sp)

    to_install: list[tuple[str, str]] = []  # (import_name, pip_name)

    for imp in sorted(imports):
        # Already available?
        if _is_importable(imp, extra_paths):
            result.already_available.append(imp)
            continue

        # Resolve PyPI name
        pip_name = await resolve_package_name(imp)
        if pip_name is None:
            result.failed.append((
                imp,
                f"Package '{imp}' not found on PyPI — it may not exist or "
                f"the import name differs from the package name.",
            ))
            continue

        to_install.append((imp, pip_name))

    if not to_install:
        return result

    # Install in parallel-ish (but sequentially to be nice to PyPI)
    Path(deps_dir).mkdir(parents=True, exist_ok=True)

    for imp, pip_name in to_install:
        log.info("Installing dependency: %s (pip: %s)", imp, pip_name)
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable, "-m", "pip", "install",
                    "--target", deps_dir,
                    "--no-warn-conflicts",
                    "--disable-pip-version-check",
                    "--quiet",
                    pip_name,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                result.installed.append(pip_name)
                log.info("Installed %s successfully", pip_name)
            else:
                err_msg = (proc.stderr or proc.stdout or "unknown error").strip()
                # Trim to something useful for the coder
                err_msg = err_msg[-200:] if len(err_msg) > 200 else err_msg
                result.failed.append((imp, f"pip install failed: {err_msg}"))
                log.warning("Failed to install %s: %s", pip_name, err_msg[:200])
        except subprocess.TimeoutExpired:
            result.failed.append((imp, "pip install timed out (>60s)"))
            log.warning("Timeout installing %s", pip_name)
        except Exception as e:
            result.failed.append((imp, f"install error: {e}"))
            log.warning("Error installing %s: %s", pip_name, e)

    return result
