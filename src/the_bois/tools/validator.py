"""Code validator — because 'it looks correct' is not a deployment strategy.

Runs actual validation on generated code:
1. Syntax check (py_compile)
2. Import check (catches NameError, ImportError at load time)
3. Test runner (pytest/unittest if test files exist)

All execution happens in a temporary directory with a subprocess timeout,
so the bois can't accidentally rm -rf / or loop forever.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── Unicode / encoding sanitisation ──────────────────────────────────
# Local models LOVE to sprinkle smart-quotes, en-dashes and other
# non-ASCII garbage into generated Python.  This map nukes them on sight.
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
    "\u2011": "-",   # non-breaking hyphen
    "\u201c": '"',   # left double smart quote
    "\u201d": '"',   # right double smart quote
    "\u2018": "'",   # left single smart quote
    "\u2019": "'",   # right single smart quote
    "\u00a0": " ",   # non-breaking space
    "\ufeff": "",    # BOM
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
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

        issues = []
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
                pytest_m = re.search(r'FAILED\s+(\S+\.py)', err)
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
                    r"'(\w+)'.*has no attribute '(\w+)'", err,
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
                    suggestion = f"Undefined name at {loc} — add the missing import or def."
            elif "ImportError" in err or "ModuleNotFoundError" in err:
                sev = "critical"
                mod_m = re.search(r"No module named '(\w+)'", err)
                if mod_m:
                    suggestion = f"Module '{mod_m.group(1)}' not found — check the import path."
                else:
                    suggestion = f"Import error at {loc} — verify the module path."
            elif "TypeError" in err:
                sev = "critical"
                suggestion = f"Type/argument error at {loc} — check the function signature."
            elif "AssertionError" in err or "FAILED" in err:
                sev = "major"
                suggestion = f"Test failure at {loc} — the assertion does not hold, check the logic."
            else:
                sev = "major"
                suggestion = f"Runtime error at {loc} — read the traceback carefully."

            issues.append({
                "severity": sev,
                "file": file_path,
                "description": err,
                "suggestion": suggestion,
            })

        return {
            "approved": False,
            "issues": issues,
            "summary": (
                "Code FAILED runtime validation. The reviewer approved the code "
                "but it crashes when actually executed. Fix the errors below."
            ),
        }


# Max wall-clock for the entire validation pass (all files combined).
# Prevents 10 files × 30s/each = 300s of just import checking.
_MAX_TOTAL_TIMEOUT = 120

# Resource limits for sandboxed subprocesses.
_RLIMIT_AS = 512 * 1024 * 1024     # 512 MB address space
_RLIMIT_CPU = 30                     # 30 seconds CPU time
_RLIMIT_FSIZE = 50 * 1024 * 1024    # 50 MB max file writes
_RLIMIT_NPROC = 50                   # max child processes (no fork bombs)

# Env vars to carry into the sandbox. Everything else is stripped.
_ENV_WHITELIST = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    "TMPDIR", "PYTHONDONTWRITEBYTECODE",
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


def _sandbox_env(tmpdir: str) -> dict[str, str]:
    """Build a minimal env for the sandbox — whitelist only safe vars."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_WHITELIST}
    env["PYTHONPATH"] = tmpdir
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
        f for f in files
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
            result.errors.append(
                f"SyntaxError in {f['path']} line {e.lineno}: {e.msg}"
            )


def _check_imports(
    py_files: list[dict],
    tmpdir: str,
    result: ValidationResult,
    timeout: int,
    overall_start: float,
) -> None:
    """Try to import non-test files in a sandbox. Mutates *result*."""
    source_files = [f for f in py_files if not _is_test_file(f["path"])]
    env = _sandbox_env(tmpdir)
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
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir, env=env,
                preexec_fn=_sandbox_preexec,
            )
            if proc.returncode != 0:
                err = _extract_traceback(proc.stderr)
                if _is_third_party_import_error(err):
                    result.errors.append(
                        f"[warning] {f['path']}: missing third-party package "
                        f"(not a code bug):\n{err}"
                    )
                else:
                    result.import_ok = False
                    result.passed = False
                    result.errors.append(
                        f"Import/load error in {f['path']}:\n{err}"
                    )
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
) -> None:
    """Run pytest/unittest on test files. Mutates *result*."""
    test_files = [f for f in py_files if _is_test_file(f["path"])]
    if not test_files:
        return

    env = _sandbox_env(tmpdir)
    python = sys.executable
    result.tests_ran = True

    proc = subprocess.run(
        [python, "-m", "pytest", "-x", "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True,
        timeout=timeout, cwd=tmpdir, env=env,
        preexec_fn=_sandbox_preexec,
    )

    if proc.returncode != 0 and "No module named pytest" in proc.stderr:
        proc = subprocess.run(
            [python, "-m", "unittest", "discover",
             "-s", ".", "-p", "test_*.py", "-v"],
            capture_output=True, text=True,
            timeout=timeout, cwd=tmpdir, env=env,
            preexec_fn=_sandbox_preexec,
        )

    if proc.returncode != 0:
        result.tests_ok = False
        result.passed = False
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        result.errors.append(f"Test failures:\n{output}")


# ── Public validation entry points ────────────────────────────────── #

def validate_fast(
    files: list[dict],
    timeout: int = 30,
) -> ValidationResult:
    """Fast validation: syntax check + import check only.

    Used between coder and reviewer to catch crashes early.
    Does NOT run tests — that's validate_full's job.
    """
    result = ValidationResult()
    tmpdir, py_files = _setup_sandbox(files)

    if not py_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    try:
        _check_syntax(py_files, tmpdir, result)
        if not result.syntax_ok:
            return result

        _check_imports(py_files, tmpdir, result, timeout, time.monotonic())
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
) -> ValidationResult:
    """Full validation: syntax + imports + test runner.

    Used after reviewer approves to verify correctness end-to-end.
    """
    result = ValidationResult()
    tmpdir, py_files = _setup_sandbox(files)

    if not py_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    try:
        _check_syntax(py_files, tmpdir, result)
        if not result.syntax_ok:
            return result

        _check_imports(py_files, tmpdir, result, timeout, time.monotonic())
        if not result.import_ok:
            return result

        _run_tests(py_files, tmpdir, result, timeout)
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
) -> ValidationResult:
    """Full validation — backwards-compatible alias for validate_full()."""
    return validate_full(files, timeout)


def _is_test_file(path: str) -> bool:
    """Check if a file path looks like a test file."""
    name = Path(path).name.lower()
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _is_third_party_import_error(traceback_text: str) -> bool:
    """Check if a traceback is just a missing third-party package.

    We don't want to fail the build because `import flask` doesn't work
    in the sandbox — that's a deployment concern, not a code bug.
    """
    indicators = ("ModuleNotFoundError", "No module named")
    lines = traceback_text.strip().split("\n")
    # The actual error is usually the last line
    if not lines:
        return False
    last_line = lines[-1].strip()
    return any(ind in last_line for ind in indicators)


def _extract_traceback(stderr: str, max_lines: int = 15) -> str:
    """Extract the most useful part of a Python traceback."""
    lines = stderr.strip().split("\n")
    if len(lines) <= max_lines:
        return stderr.strip()

    # Find the last "Traceback" line and take everything after it
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Traceback"):
            return "\n".join(lines[i:i + max_lines])

    # Fallback: last N lines
    return "\n".join(lines[-max_lines:])
