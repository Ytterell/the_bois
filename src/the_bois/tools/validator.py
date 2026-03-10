"""Code validator — because 'it looks correct' is not a deployment strategy.

Runs actual validation on generated code:
1. Syntax check (py_compile)
2. Import check (catches NameError, ImportError at load time)
3. Test runner (pytest/unittest if test files exist)

All execution happens in a temporary directory with a subprocess timeout,
so the bois can't accidentally rm -rf / or loop forever.
"""

from __future__ import annotations
import asyncio
import logging

import os
import resource
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ── Unicode / encoding sanitisation ──────────────────────────────────
# Local models LOVE to sprinkle smart-quotes, en-dashes and other
# non-ASCII garbage into generated Python.  This map nukes them on sight.
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
    "\u2011": "-",  # non-breaking hyphen
    "\u201c": '"',  # left double smart quote
    "\u201d": '"',  # right double smart quote
    "\u2018": "'",  # left single smart quote
    "\u2019": "'",  # right single smart quote
    "\u00a0": " ",  # non-breaking space
    "\ufeff": "",  # BOM
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
}


def sanitize_code(content: str) -> str:
    """Replace common Unicode garbage with ASCII equivalents.

    Call this on every piece of generated code BEFORE validation.
    It's cheaper than one more round-trip to the model.
    """
    for bad, good in _UNICODE_REPLACEMENTS.items():
        content = content.replace(bad, good)
    return content


@dataclass
class ValidationResult:
    """Structured result from code validation."""

    passed: bool = True
    syntax_ok: bool = True
    import_ok: bool = True
    tests_ran: bool = False
    tests_ok: bool = True
    errors: list[str] = field(default_factory=list)

    def as_feedback(self) -> dict:
        """Format as reviewer-style feedback for the coder."""
        import re

        from the_bois.contracts import ReviewFeedback, ReviewIssue

        issues: list[ReviewIssue] = []
        for err in self.errors:
            # ── Extract file path and line number from traceback ──
            file_path = "runtime"
            line_no = ""
            # Standard traceback: File "foo.py", line 42
            tb_m = re.search(r'File "(.+?)"(?:,\s*line\s*(\d+))?', err)
            if tb_m:
                file_path = tb_m.group(1)
                line_no = tb_m.group(2) or ""
            # pytest short: FAILED test.py::test_foo
            if file_path == "runtime":
                pytest_m = re.search(r"FAILED\s+(\S+\.py)", err)
                if pytest_m:
                    file_path = pytest_m.group(1)

            loc = f"{file_path}:{line_no}" if line_no else file_path

            # ── Severity + targeted suggestion per error type ──
            if "SyntaxError" in err:
                sev = "critical"
                suggestion = f"Fix syntax error at {loc}."
            elif "AttributeError" in err:
                sev = "critical"
                attr_m = re.search(
                    r"'(\w+)'.*has no attribute '(\w+)'",
                    err,
                )
                if attr_m:
                    cls, attr = attr_m.group(1), attr_m.group(2)
                    suggestion = (
                        f"'{attr}' does not exist on '{cls}' — check the "
                        f"library docs for the correct attribute/method name."
                    )
                else:
                    suggestion = f"Wrong attribute access at {loc} — verify the object's actual API."
            elif "NameError" in err:
                sev = "critical"
                name_m = re.search(r"name '(\w+)' is not defined", err)
                if name_m:
                    suggestion = (
                        f"'{name_m.group(1)}' is not defined — add the "
                        f"missing import or function definition."
                    )
                else:
                    suggestion = (
                        f"Undefined name at {loc} — add the missing import or def."
                    )
            elif "ImportError" in err or "ModuleNotFoundError" in err:
                sev = "critical"
                mod_m = re.search(r"No module named '(\w+)'", err)
                if mod_m:
                    suggestion = (
                        f"Module '{mod_m.group(1)}' not found — check the import path."
                    )
                else:
                    suggestion = f"Import error at {loc} — verify the module path."
            elif "TypeError" in err:
                sev = "critical"
                suggestion = (
                    f"Type/argument error at {loc} — check the function signature."
                )
            elif (
                "Ran 0 tests" in err
                or "NO TESTS RAN" in err
                or "no tests ran" in err.lower()
            ):
                sev = "critical"
                suggestion = (
                    "Zero tests were discovered by the test runner. Common causes: "
                    "1) pytest not available in sandbox — use unittest.TestCase with "
                    "setUp/tearDown instead of @pytest.fixture and pytest-specific "
                    "features like tmp_path, "
                    "2) test files not matching discovery pattern (must be test_*.py), "
                    "3) test functions not prefixed with test_, "
                    "4) ImportError in test module preventing load — check that all "
                    "imports in test files resolve correctly. "
                    "REWRITE tests using unittest.TestCase if you used pytest features."
                )
            elif "Test discovery diagnostic" in err:
                sev = "critical"
                suggestion = (
                    "A test file failed to import during discovery. Fix the import "
                    "error shown above — this is why zero tests ran."
                )
            elif "AssertionError" in err or "FAILED" in err:
                sev = "major"
                suggestion = f"Test failure at {loc} — the assertion does not hold, check the logic."
            else:
                sev = "major"
                suggestion = f"Runtime error at {loc} — read the traceback carefully."

            issues.append(
                ReviewIssue(
                    severity=sev,
                    file=file_path,
                    description=err,
                    suggestion=suggestion,
                )
            )

        return ReviewFeedback(
            approved=False,
            issues=issues,
            summary=(
                "Code FAILED runtime validation. The reviewer approved the code "
                "but it crashes when actually executed. Fix the errors below."
            ),
        ).to_dict()


# Max wall-clock for the entire validation pass (all files combined).
# Prevents 10 files × 30s/each = 300s of just import checking.
_MAX_TOTAL_TIMEOUT = 120

# Resource limits for sandboxed subprocesses.
_RLIMIT_AS = 512 * 1024 * 1024  # 512 MB address space
_RLIMIT_CPU = 30  # 30 seconds CPU time
_RLIMIT_FSIZE = 50 * 1024 * 1024  # 50 MB max file writes
_RLIMIT_NPROC = 50  # max child processes (no fork bombs)

# Env vars to carry into the sandbox. Everything else is stripped.
_ENV_WHITELIST = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "PYTHONDONTWRITEBYTECODE",
}


def _sandbox_preexec() -> None:
    """Pre-exec hook for subprocess: set resource limits.

    Runs in the child process before exec, so limits only apply to
    the sandboxed code, not to the orchestrator itself.
    """
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_RLIMIT_AS, _RLIMIT_AS))
    except (ValueError, OSError):
        pass  # macOS may report differently but still enforces
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_RLIMIT_CPU, _RLIMIT_CPU))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE, _RLIMIT_FSIZE))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (_RLIMIT_NPROC, _RLIMIT_NPROC))
    except (ValueError, OSError):
        pass


def _get_venv_site_packages() -> str | None:
    """Detect the active venv's site-packages directory.

    Returns the path if we're running inside a venv (or if we can find
    a .venv next to the project root), otherwise None.
    """
    # Case 1: the_bois itself is running in a venv
    if sys.prefix != sys.base_prefix:
        sp = sysconfig.get_path("purelib")
        if sp and Path(sp).is_dir():
            return sp

    # Case 2: look for a .venv in the project root (common convention)
    # Walk up from this file to find the project root (contains pyproject.toml)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            for venv_name in (".venv", "venv"):
                venv_dir = parent / venv_name
                if venv_dir.is_dir():
                    # Find the site-packages inside this venv
                    for sp in sorted(venv_dir.glob("lib/python*/site-packages")):
                        if sp.is_dir():
                            return str(sp)
            break  # found project root but no venv

    return None


def _sandbox_env(
    tmpdir: str,
    inherit_venv: bool = True,
    deps_dir: str | None = None,
) -> dict[str, str]:
    """Build a minimal env for the sandbox — whitelist only safe vars.

    When *inherit_venv* is True (default), the project venv's
    site-packages are added to PYTHONPATH so third-party libraries
    (pytest, textual, etc.) are available for import checking and
    test execution.  This lets the validator catch *wrong* imports
    (e.g. ``from textual.widgets import List``) instead of blanket-
    warning on every third-party import.

    When *deps_dir* is set, auto-installed dependency packages are
    also added to PYTHONPATH (before venv, so fresh installs win).
    """
    env = {k: v for k, v in os.environ.items() if k in _ENV_WHITELIST}
    python_paths = [tmpdir]

    if deps_dir and Path(deps_dir).is_dir():
        python_paths.append(deps_dir)
        log.debug("Sandbox using deps_dir: %s", deps_dir)

    if inherit_venv:
        venv_sp = _get_venv_site_packages()
        if venv_sp:
            python_paths.append(venv_sp)
            log.debug("Sandbox inheriting venv site-packages: %s", venv_sp)

    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["HOME"] = tmpdir  # don't let code read real home
    return env


def _setup_sandbox(files: list[dict]) -> tuple[str, list[dict]]:
    """Shared sandbox setup: sanitise, write files, return (tmpdir, py_files)."""
    for f in files:
        if f.get("content"):
            f["content"] = sanitize_code(f["content"])

    tmpdir = tempfile.mkdtemp(prefix="the_bois_validate_")
    for f in files:
        fpath = Path(tmpdir) / f["path"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(f.get("content", ""))

    py_files = [
        f
        for f in files
        if f.get("path", "").endswith(".py") and f.get("content", "").strip()
    ]
    return tmpdir, py_files


def _check_syntax(py_files: list[dict], tmpdir: str, result: ValidationResult) -> None:
    """Run py_compile on all Python files. Mutates *result* in place."""
    for f in py_files:
        fpath = Path(tmpdir) / f["path"]
        try:
            compile(fpath.read_text(), f["path"], "exec")
        except SyntaxError as e:
            result.syntax_ok = False
            result.passed = False
            result.errors.append(f"SyntaxError in {f['path']} line {e.lineno}: {e.msg}")


def _check_imports(
    py_files: list[dict],
    tmpdir: str,
    result: ValidationResult,
    timeout: int,
    overall_start: float,
    deps_dir: str | None = None,
    deps_installed: bool = False,
) -> None:
    """Try to import non-test files in a sandbox. Mutates *result*."""
    source_files = [f for f in py_files if not _is_test_file(f["path"])]
    env = _sandbox_env(tmpdir, deps_dir=deps_dir)
    python = sys.executable

    for f in source_files:
        if time.monotonic() - overall_start > _MAX_TOTAL_TIMEOUT:
            result.errors.append(
                f"Overall validation timeout ({_MAX_TOTAL_TIMEOUT}s) exceeded. "
                f"Skipping remaining import checks."
            )
            result.import_ok = False
            result.passed = False
            return

        fpath = Path(tmpdir) / f["path"]
        import_script = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('_mod', r'{fpath}'); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod); "
            "print('OK')"
        )
        try:
            proc = subprocess.run(
                [python, "-c", import_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
                preexec_fn=_sandbox_preexec,
            )
            if proc.returncode != 0:
                err = _extract_traceback(proc.stderr)
                if _is_third_party_import_error(err, deps_installed=deps_installed):
                    result.errors.append(
                        f"[warning] {f['path']}: missing third-party package "
                        f"(not a code bug):\n{err}"
                    )
                else:
                    result.import_ok = False
                    result.passed = False
                    result.errors.append(f"Import/load error in {f['path']}:\n{err}")
        except subprocess.TimeoutExpired:
            result.errors.append(
                f"Timeout importing {f['path']} (>{timeout}s) — "
                f"module may have blocking code at module level."
            )
            result.import_ok = False
            result.passed = False


def _run_tests(
    py_files: list[dict],
    tmpdir: str,
    result: ValidationResult,
    timeout: int,
    deps_dir: str | None = None,
) -> None:
    """Run pytest/unittest on test files. Mutates *result*."""
    test_files = [f for f in py_files if _is_test_file(f["path"])]
    if not test_files:
        return

    env = _sandbox_env(tmpdir, deps_dir=deps_dir)
    python = sys.executable
    result.tests_ran = True

    proc = subprocess.run(
        [python, "-m", "pytest", "-x", "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=tmpdir,
        env=env,
        preexec_fn=_sandbox_preexec,
    )

    if proc.returncode != 0 and "No module named pytest" in proc.stderr:
        proc = subprocess.run(
            [python, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            env=env,
            preexec_fn=_sandbox_preexec,
        )

    if proc.returncode != 0:
        result.tests_ok = False
        result.passed = False
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        result.errors.append(f"Test failures:\n{output}")

    # ── Diagnose "0 tests ran" — try importing each test file to find why ──
    combined_output = proc.stdout + "\n" + proc.stderr
    if "Ran 0 tests" in combined_output or "no tests ran" in combined_output.lower():
        for tf in test_files:
            tf_path = Path(tmpdir) / tf["path"]
            module_name = tf["path"].replace("/", ".").replace(".py", "")
            diag_proc = subprocess.run(
                [
                    python,
                    "-c",
                    f"import importlib; importlib.import_module('{module_name}')",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
                env=env,
                preexec_fn=_sandbox_preexec,
            )
            if diag_proc.returncode != 0:
                diag_err = _extract_traceback(diag_proc.stderr, max_lines=8)
                result.errors.append(
                    f"Test discovery diagnostic: {tf['path']} failed to "
                    f"import:\n{diag_err}"
                )


# ── Dependency auto-install bridge ──────────────────────────────────── #


def _ensure_sandbox_deps(
    files: list[dict],
    deps_dir: str,
    result: ValidationResult,
) -> bool:
    """Extract third-party imports and install missing packages.

    Returns True if the install step ran (regardless of per-package
    success), so callers know that top-level import failures are now
    real errors rather than sandbox limitations.

    Appends DEPENDENCY ERROR entries to *result.errors* for packages
    that couldn't be installed.
    """
    from the_bois.tools.deps import extract_imports, ensure_deps

    imports = extract_imports(files)
    if not imports:
        return False

    log.info("Detected third-party imports: %s", sorted(imports))

    venv_sp = _get_venv_site_packages()

    # ensure_deps is async — run it from sync context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (the orchestrator's event loop).
        # Can't await directly from a sync function, so use a thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            deps_result = pool.submit(
                asyncio.run, ensure_deps(imports, deps_dir, venv_sp)
            ).result(timeout=300)
    else:
        deps_result = asyncio.run(ensure_deps(imports, deps_dir, venv_sp))

    if deps_result.installed:
        log.info("Installed deps: %s", deps_result.installed)
    if deps_result.already_available:
        log.debug("Already available: %s", deps_result.already_available)
    if deps_result.failed:
        for pkg_name, err_msg in deps_result.failed:
            result.errors.append(
                f"DEPENDENCY ERROR: Package '{pkg_name}' could not be "
                f"installed: {err_msg}. Use a different package or check "
                f"the import name."
            )
            result.passed = False
        log.warning("Failed deps: %s", deps_result.failed)

    return True  # deps install was attempted


# ── Public validation entry points ──────────────────────────────────────── #


def validate_fast(
    files: list[dict],
    timeout: int = 30,
    deps_dir: str | None = None,
) -> ValidationResult:
    """Fast validation: syntax check + import check only.

    Used between coder and reviewer to catch crashes early.
    Does NOT run tests — that's validate_full's job.

    When *deps_dir* is set, auto-installs missing third-party packages
    before running import checks, so we get real feedback instead of
    blanket "missing package" warnings.
    """
    result = ValidationResult()
    tmpdir, py_files = _setup_sandbox(files)

    if not py_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    deps_installed = False
    try:
        _check_syntax(py_files, tmpdir, result)
        if not result.syntax_ok:
            return result

        # ── Auto-install deps before import checking ──
        if deps_dir:
            deps_installed = _ensure_sandbox_deps(files, deps_dir, result)

        _check_imports(
            py_files,
            tmpdir,
            result,
            timeout,
            time.monotonic(),
            deps_dir=deps_dir,
            deps_installed=deps_installed,
        )
        return result

    except Exception as e:
        result.passed = False
        result.errors.append(f"Validation infrastructure error: {e}")
        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def validate_full(
    files: list[dict],
    timeout: int = 30,
    deps_dir: str | None = None,
    fast_passed: bool = False,
) -> ValidationResult:
    """Full validation: syntax + imports + test runner.

    Used after reviewer approves to verify correctness end-to-end.
    When *deps_dir* is set, deps are already installed from validate_fast.

    When *fast_passed* is True, syntax and import checks are skipped
    because validate_fast() already verified them on the same code.
    This avoids redundant work and redundant dep-install attempts.
    """
    result = ValidationResult()
    tmpdir, py_files = _setup_sandbox(files)

    if not py_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    deps_installed = False
    try:
        if fast_passed:
            # validate_fast already confirmed syntax + imports + deps.
            # Mark those as passing and skip straight to tests.
            result.syntax_ok = True
            result.import_ok = True
            deps_installed = bool(deps_dir)
        else:
            _check_syntax(py_files, tmpdir, result)
            if not result.syntax_ok:
                return result

            # ── Auto-install deps (in case validate_fast wasn't called) ──
            if deps_dir:
                deps_installed = _ensure_sandbox_deps(files, deps_dir, result)

            _check_imports(
                py_files,
                tmpdir,
                result,
                timeout,
                time.monotonic(),
                deps_dir=deps_dir,
                deps_installed=deps_installed,
            )
            if not result.import_ok:
                return result

        _run_tests(py_files, tmpdir, result, timeout, deps_dir=deps_dir)
        return result

    except Exception as e:
        result.passed = False
        result.errors.append(f"Validation infrastructure error: {e}")
        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def validate_code(
    files: list[dict],
    timeout: int = 30,
    deps_dir: str | None = None,
) -> ValidationResult:
    """Full validation — backwards-compatible alias for validate_full()."""
    return validate_full(files, timeout, deps_dir=deps_dir)


def _is_test_file(path: str) -> bool:
    """Check if a file path looks like a test file."""
    name = Path(path).name.lower()
    return (
        name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def _is_third_party_import_error(
    traceback_text: str,
    deps_installed: bool = False,
) -> bool:
    """Check if a traceback is just a missing third-party package.

    We don't want to fail the build because ``import flask`` doesn't
    work in the sandbox — that's a deployment concern, not a code bug.

    BUT: ``cannot import name 'List' from 'textual.widgets'`` means the
    package IS installed and the import path is wrong — that IS a bug.
    Same for ``No module named 'textual.widgets.bogus'`` (sub-module
    not found inside an installed package).

    When *deps_installed* is True, we've already tried to install all
    detected third-party packages.  A top-level ModuleNotFoundError
    now means the package genuinely doesn't exist on PyPI — that's a
    real error the coder needs to fix, not a sandbox limitation.
    """
    import re

    lines = traceback_text.strip().split("\n")
    if not lines:
        return False
    last_line = lines[-1].strip()

    # "cannot import name 'X' from 'Y'" → package exists, name is wrong
    if "cannot import name" in last_line:
        return False

    # "No module named 'foo.bar.baz'" → sub-module of an installed package
    # vs. "No module named 'foo'" → top-level package missing entirely
    mod_m = re.search(r"No module named ['\"]([^'\"]+)['\"]", last_line)
    if mod_m:
        module_path = mod_m.group(1)
        if "." in module_path:
            return False  # treat as potential code bug
        # Top-level module missing.  If deps were installed, this is real.
        if deps_installed:
            return False  # treat as real error — pip already tried
        return True

    # ModuleNotFoundError without a parseable module name
    if "ModuleNotFoundError" in last_line:
        if deps_installed:
            return False  # real error after deps install attempt
        return True

    return False


def _extract_traceback(stderr: str, max_lines: int = 15) -> str:
    """Extract the most useful part of a Python traceback."""
    lines = stderr.strip().split("\n")
    if len(lines) <= max_lines:
        return stderr.strip()

    # Find the last "Traceback" line and take everything after it
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Traceback"):
            return "\n".join(lines[i : i + max_lines])

    # Fallback: last N lines
    return "\n".join(lines[-max_lines:])
