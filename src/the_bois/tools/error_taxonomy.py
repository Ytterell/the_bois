"""Error taxonomy and targeted recovery strategies.

Classifies validation/review errors into categories and returns
per-category retry strategies so the orchestrator makes *smart*
decisions instead of blindly bumping temperature and hoping for the best.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from the_bois.tools.validator import ValidationResult

# ── Error categories ────────────────────────────────────────────────── #


class ErrorCategory(str, Enum):
    """Taxonomy of things that go wrong in generated code."""

    SYNTAX = "syntax"
    IMPORT_MISSING = "import_missing"  # ModuleNotFoundError
    IMPORT_BAD_PATH = "import_bad_path"  # ImportError — wrong submodule/name
    NAME_ERROR = "name_error"  # Undefined variable / function
    ATTRIBUTE_ERROR = "attribute_error"  # Wrong API name on an object
    TYPE_ERROR = "type_error"  # Wrong arguments / types
    TEST_FAILURE = "test_failure"  # Assertion / test logic wrong
    TEST_DISCOVERY = "test_discovery"  # Zero tests ran
    DEPENDENCY = "dependency"  # Package doesn't exist on PyPI
    TIMEOUT = "timeout"  # Resource/time limit exceeded
    RESOURCE = "resource"  # Memory / process limits
    LOGIC = "logic"  # Reviewer-identified logic error
    SIGNATURE_MISMATCH = "signature_mismatch"  # Required signatures missing
    UNKNOWN = "unknown"


# ── Retry strategy per category ─────────────────────────────────────── #


@dataclass
class RetryStrategy:
    """What the orchestrator should do differently on retry."""

    temperature_delta: float = 0.0
    should_research: bool = False
    should_simplify: bool = False
    retry_hint: str = ""


# Strategy lookup — the core of the taxonomy.
# temperature_delta: 0 = keep base, positive = more creative, negative = more precise
_STRATEGIES: dict[ErrorCategory, RetryStrategy] = {
    ErrorCategory.SYNTAX: RetryStrategy(
        temperature_delta=-0.1,
        retry_hint=(
            "Your code has a SYNTAX ERROR. Lower your creativity — just write "
            "correct Python. Read the error message carefully and fix the exact line."
        ),
    ),
    ErrorCategory.IMPORT_MISSING: RetryStrategy(
        should_research=True,
        retry_hint=(
            "A module you imported does not exist. Use NEEDS_RESEARCH to find "
            "the correct package, or switch to a stdlib-only approach."
        ),
    ),
    ErrorCategory.IMPORT_BAD_PATH: RetryStrategy(
        should_research=True,
        retry_hint=(
            "Your import path is wrong — the module exists but you're importing "
            "from the wrong submodule. Check the API REFERENCE or use "
            "NEEDS_RESEARCH to find the correct import path."
        ),
    ),
    ErrorCategory.NAME_ERROR: RetryStrategy(
        temperature_delta=-0.05,
        retry_hint=(
            "You referenced a name that doesn't exist. Either you forgot to "
            "import it, forgot to define it, or there's a typo. Check every "
            "name you use against your imports and definitions."
        ),
    ),
    ErrorCategory.ATTRIBUTE_ERROR: RetryStrategy(
        should_research=True,
        retry_hint=(
            "You used an attribute/method that doesn't exist on the object. "
            "Do NOT guess API names — only use names from the API REFERENCE. "
            "If no reference is available, use NEEDS_RESEARCH."
        ),
    ),
    ErrorCategory.TYPE_ERROR: RetryStrategy(
        retry_hint=(
            "Wrong argument types or count in a function call. Check the "
            "function signature carefully — you may be passing the wrong "
            "number of arguments or the wrong types."
        ),
    ),
    ErrorCategory.TEST_FAILURE: RetryStrategy(
        temperature_delta=0.15,
        retry_hint=(
            "Tests run but assertions fail. Re-read the requirements and "
            "trace through your logic step by step. The test expectations "
            "are correct — your implementation has a bug."
        ),
    ),
    ErrorCategory.TEST_DISCOVERY: RetryStrategy(
        retry_hint=(
            "ZERO tests were discovered. The sandbox may not have pytest. "
            "REWRITE all tests using unittest.TestCase with setUp/tearDown. "
            "Do NOT use @pytest.fixture, tmp_path, or pytest-specific features. "
            "Test files must be named test_*.py with test_ prefixed methods."
        ),
    ),
    ErrorCategory.DEPENDENCY: RetryStrategy(
        should_research=True,
        retry_hint=(
            "A package you need doesn't exist on PyPI. The import name "
            "probably differs from the pip package name. Use NEEDS_RESEARCH "
            "to find the correct package, or use a stdlib-only alternative."
        ),
    ),
    ErrorCategory.TIMEOUT: RetryStrategy(
        should_simplify=True,
        retry_hint=(
            "Your code exceeded the time limit. Simplify your approach — "
            "avoid infinite loops, reduce iteration counts, and add timeout "
            "handling to any network/IO operations."
        ),
    ),
    ErrorCategory.RESOURCE: RetryStrategy(
        should_simplify=True,
        retry_hint=(
            "Your code exceeded memory or process limits. Use less memory — "
            "avoid loading large datasets into memory at once, use generators "
            "instead of lists where possible."
        ),
    ),
    ErrorCategory.LOGIC: RetryStrategy(
        temperature_delta=0.15,
        retry_hint=(
            "The reviewer found a logic error. Re-read the task requirements "
            "carefully and think through edge cases. Your code runs but "
            "produces wrong results."
        ),
    ),
    ErrorCategory.SIGNATURE_MISMATCH: RetryStrategy(
        temperature_delta=-0.05,
        retry_hint=(
            "Required function signatures are missing. The architect specified "
            "exact signatures that MUST exist. Implement them."
        ),
    ),
    ErrorCategory.UNKNOWN: RetryStrategy(
        temperature_delta=0.1,
        retry_hint="An unexpected error occurred. Read the traceback carefully.",
    ),
}


# ── Classification engine ───────────────────────────────────────────── #


@dataclass
class ClassifiedError:
    """An error string with its determined category."""

    raw: str
    category: ErrorCategory
    extracted_name: str = ""  # e.g. the missing attribute, module, etc.


def classify_error(error: str) -> ClassifiedError:
    """Classify a single error string into a category.

    Examines the error text for known patterns (exception names,
    keywords) and returns the best-matching category.
    """
    e = error  # shorthand

    # ── Syntax ──
    if "SyntaxError" in e:
        return ClassifiedError(e, ErrorCategory.SYNTAX)

    # ── Dependency (check before import errors — more specific) ──
    if "DEPENDENCY ERROR" in e:
        m = re.search(r"Package '(\w+)'", e)
        return ClassifiedError(e, ErrorCategory.DEPENDENCY, m.group(1) if m else "")

    # ── Import errors ──
    if "ModuleNotFoundError" in e or "No module named" in e:
        m = re.search(r"No module named '([\w.]+)'", e)
        return ClassifiedError(
            e,
            ErrorCategory.IMPORT_MISSING,
            m.group(1) if m else "",
        )
    if "ImportError" in e or "cannot import name" in e:
        m = re.search(r"cannot import name '(\w+)'", e)
        return ClassifiedError(
            e,
            ErrorCategory.IMPORT_BAD_PATH,
            m.group(1) if m else "",
        )

    # ── Name/Attribute/Type ──
    if "NameError" in e:
        m = re.search(r"name '(\w+)' is not defined", e)
        return ClassifiedError(e, ErrorCategory.NAME_ERROR, m.group(1) if m else "")

    if "AttributeError" in e:
        m = re.search(r"has no attribute '(\w+)'", e)
        return ClassifiedError(
            e,
            ErrorCategory.ATTRIBUTE_ERROR,
            m.group(1) if m else "",
        )

    if "TypeError" in e:
        return ClassifiedError(e, ErrorCategory.TYPE_ERROR)

    # ── Test issues ──
    if "Ran 0 tests" in e or "NO TESTS RAN" in e or "no tests ran" in e.lower():
        return ClassifiedError(e, ErrorCategory.TEST_DISCOVERY)

    if "Test discovery diagnostic" in e:
        return ClassifiedError(e, ErrorCategory.TEST_DISCOVERY)

    if "FAILED" in e or "AssertionError" in e or "AssertionError" in e:
        return ClassifiedError(e, ErrorCategory.TEST_FAILURE)

    # ── Resource / timeout ──
    if "timed out" in e.lower() or "TimeoutExpired" in e or "RLIMIT_CPU" in e:
        return ClassifiedError(e, ErrorCategory.TIMEOUT)

    if "MemoryError" in e or "RLIMIT_AS" in e or "Cannot allocate" in e:
        return ClassifiedError(e, ErrorCategory.RESOURCE)

    # ── Signature (from orchestrator's static checker) ──
    if (
        "MISSING SIGNATURES" in e
        or "signature" in e.lower()
        and "required" in e.lower()
    ):
        return ClassifiedError(e, ErrorCategory.SIGNATURE_MISMATCH)

    return ClassifiedError(e, ErrorCategory.UNKNOWN)


def classify_errors(errors: list[str]) -> list[ClassifiedError]:
    """Classify a list of error strings."""
    return [classify_error(e) for e in errors]


def classify_validation_result(result: ValidationResult) -> list[ClassifiedError]:
    """Classify all errors from a ValidationResult."""
    return classify_errors(result.errors)


# ── Strategy resolution ──────────────────────────────────────────────── #


def get_strategy(category: ErrorCategory) -> RetryStrategy:
    """Get the retry strategy for an error category."""
    return _STRATEGIES.get(category, _STRATEGIES[ErrorCategory.UNKNOWN])


def get_dominant_strategy(errors: list[ClassifiedError]) -> RetryStrategy:
    """Determine the best retry strategy from a set of classified errors.

    When multiple error types are present, picks the strategy that best
    addresses the *most critical* error.  Priority:
      1. SYNTAX (nothing else matters if it won't parse)
      2. IMPORT_* / DEPENDENCY (can't run if deps are broken)
      3. NAME_ERROR / ATTRIBUTE_ERROR / TYPE_ERROR (runtime crashes)
      4. TEST_DISCOVERY (tests can't even start)
      5. TEST_FAILURE (tests run but fail)
      6. Everything else

    Returns the strategy for the highest-priority category found.
    """
    if not errors:
        return _STRATEGIES[ErrorCategory.UNKNOWN]

    _PRIORITY: list[ErrorCategory] = [
        ErrorCategory.SYNTAX,
        ErrorCategory.DEPENDENCY,
        ErrorCategory.IMPORT_MISSING,
        ErrorCategory.IMPORT_BAD_PATH,
        ErrorCategory.NAME_ERROR,
        ErrorCategory.ATTRIBUTE_ERROR,
        ErrorCategory.TYPE_ERROR,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RESOURCE,
        ErrorCategory.SIGNATURE_MISMATCH,
        ErrorCategory.TEST_DISCOVERY,
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.LOGIC,
        ErrorCategory.UNKNOWN,
    ]

    categories_present = {e.category for e in errors}
    for cat in _PRIORITY:
        if cat in categories_present:
            return _STRATEGIES[cat]

    return _STRATEGIES[ErrorCategory.UNKNOWN]


def build_taxonomy_retry_hint(classified: list[ClassifiedError]) -> str:
    """Build a targeted retry hint from classified errors.

    Combines the dominant strategy's hint with specific details
    extracted from each error (module names, attribute names, etc.).
    """
    if not classified:
        return ""

    strategy = get_dominant_strategy(classified)
    parts: list[str] = [strategy.retry_hint]

    # Add specific details for the most actionable errors
    seen_names: set[str] = set()
    for err in classified:
        if err.extracted_name and err.extracted_name not in seen_names:
            seen_names.add(err.extracted_name)
            if err.category == ErrorCategory.ATTRIBUTE_ERROR:
                parts.append(
                    f"Specifically: '{err.extracted_name}' does not exist on that object."
                )
            elif err.category == ErrorCategory.IMPORT_MISSING:
                parts.append(
                    f"Specifically: module '{err.extracted_name}' was not found."
                )
            elif err.category == ErrorCategory.NAME_ERROR:
                parts.append(
                    f"Specifically: '{err.extracted_name}' is not defined anywhere."
                )
            elif err.category == ErrorCategory.DEPENDENCY:
                parts.append(
                    f"Specifically: package '{err.extracted_name}' does not exist on PyPI."
                )

    return "\n".join(parts)
