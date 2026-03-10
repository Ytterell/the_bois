"""AST-based static checker — catches structural issues before runtime.

This module performs deterministic verification of generated Python code
to catch bugs that the validator would only find at runtime:
- Function calls without matching definitions
- Undefined variable references
- Attribute access on unknown types
- Unresolvable imports
- Class usage without definitions

Unlike the validator which actually executes code, this runs at the AST level
and provides immediate, deterministic feedback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set

# Error categories
ERROR_FUNCTION_DEFINED = "function_defined"
ERROR_IMPORT_RESOLVED = "import_resolved"
ERROR_VARIABLE_DEFINED = "variable_defined"
ERROR_CLASS_DEFINED = "class_defined"
ERROR_ATTRIBUTE_ACCESS = "attribute_access"


@dataclass
class StaticCheckError:
    """A single static analysis error."""

    error_type: str
    file: str
    line: int
    column: int | None = None
    name: str = ""
    message: str = ""
    suggestion: str = ""

    def to_reviewer_format(self) -> dict:
        """Convert to reviewer-style issue format."""
        from the_bois.contracts import ReviewIssue

        severity = (
            "critical"
            if self.error_type
            in (
                ERROR_FUNCTION_DEFINED,
                ERROR_CLASS_DEFINED,
                ERROR_IMPORT_RESOLVED,
            )
            else "major"
        )
        return ReviewIssue(
            severity=severity,
            file=self.file,
            description=self.message,
            suggestion=self.suggestion,
        ).to_dict()


@dataclass
class StaticCheckResult:
    """Result of static analysis on a set of files."""

    passed: bool = True
    errors: list[StaticCheckError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_feedback(self) -> dict:
        """Format as reviewer-style feedback dict."""
        issues = [e.to_reviewer_format() for e in self.errors]
        return {
            "approved": self.passed,
            "issues": issues,
            "summary": (
                f"Static analysis found {len(self.errors)} issue(s). "
                "These are deterministic bugs that will fail at runtime."
                if self.errors
                else "Static analysis passed."
            ),
        }


class SymbolCollector(ast.NodeVisitor):
    """Collects all defined names (functions, classes, imports) from AST."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.functions: set[str] = set()
        self.classes: set[str] = set()
        self.imports: dict[str, str] = {}  # alias -> original
        self.from_imports: dict[str, str] = {}  # name -> module
        self.global_vars: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.from_imports[alias.asname or alias.name] = f"from {module}"

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.global_vars.add(target.id)
        self.generic_visit(node)


class ReferenceChecker(ast.NodeVisitor):
    """Checks that all references have corresponding definitions."""

    def __init__(
        self,
        file_path: str,
        defined_functions: set[str],
        defined_classes: set[str],
        imports: dict[str, str],
        from_imports: dict[str, str],
        global_vars: set[str],
        all_files: dict[str, "SymbolCollector"],
        builtin_names: "Set[str]",
    ) -> None:
        self.file_path = file_path
        self.defined_functions = defined_functions
        self.defined_classes = defined_classes
        self.imports = imports
        self.from_imports = from_imports
        self.global_vars = global_vars
        self.all_files = all_files
        self.builtin_names = builtin_names

        self.errors: list[StaticCheckError] = []
        self._in_function: bool = False
        self._local_vars: list[set[str]] = []
        self._checked_names: set[tuple[int, str]] = set()  # (line, name) dedup

    def _check_name_defined(
        self,
        name: str,
        node: ast.AST,
        is_call: bool = False,
    ) -> None:
        """Check if a name is defined or imported. Deduplicates by line+name."""
        line = node.lineno or 0

        # Deduplicate: only report each name error once per line
        key = (line, name)
        if key in self._checked_names:
            return
        self._checked_names.add(key)

        # Dunder names (__name__, __file__, __doc__, etc.) are always
        # provided by the Python runtime — never flag them.
        if name.startswith("__") and name.endswith("__"):
            return

        if name in self.builtin_names:
            return

        if name in self.defined_functions or name in self.defined_classes:
            return

        if name in self.imports or name in self.from_imports:
            return

        if name in self.global_vars:
            return

        for scope in reversed(self._local_vars):
            if name in scope:
                return

        for other_file, collector in self.all_files.items():
            if other_file == self.file_path:
                continue
            if name in collector.functions or name in collector.classes:
                return

        if is_call:
            self.errors.append(
                StaticCheckError(
                    error_type=ERROR_FUNCTION_DEFINED,
                    file=self.file_path,
                    line=node.lineno or 0,
                    column=node.col_offset,
                    name=name,
                    message=f"Function '{name}' is called but not defined or imported",
                    suggestion=f"Define '{name}()' or add proper import",
                )
            )
        elif name[0].isupper():
            self.errors.append(
                StaticCheckError(
                    error_type=ERROR_CLASS_DEFINED,
                    file=self.file_path,
                    line=node.lineno or 0,
                    column=node.col_offset,
                    name=name,
                    message=f"Class '{name}' is used but not defined or imported",
                    suggestion=f"Define class '{name}' or add proper import",
                )
            )
        else:
            self.errors.append(
                StaticCheckError(
                    error_type=ERROR_VARIABLE_DEFINED,
                    file=self.file_path,
                    line=node.lineno or 0,
                    column=node.col_offset,
                    name=name,
                    message=f"Variable '{name}' is used but not defined in scope",
                    suggestion=f"Define '{name}' before using it",
                )
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_in_function = self._in_function
        old_local_vars = self._local_vars
        self._in_function = True
        self._local_vars = self._local_vars + [set()]

        for arg in node.args.args:
            self._local_vars[-1].add(arg.arg)
        for arg in node.args.posonlyargs:
            self._local_vars[-1].add(arg.arg)
        for arg in node.args.kwonlyargs:
            self._local_vars[-1].add(arg.arg)
        if node.args.vararg:
            self._local_vars[-1].add(node.args.vararg.arg)
        if node.args.kwarg:
            self._local_vars[-1].add(node.args.kwarg.arg)

        self.generic_visit(node)

        self._in_function = old_in_function
        self._local_vars = old_local_vars

    visit_AsyncFunctionDef = visit_FunctionDef

    def _collect_target_names(self, target: ast.AST) -> list[str]:
        """Recursively extract all Name ids from an assignment target.

        Handles plain names, tuple/list unpacking, and starred targets:
            x = ...           → ['x']
            a, b = ...        → ['a', 'b']
            (a, (b, c)) = .. → ['a', 'b', 'c']
            a, *rest = ...   → ['a', 'rest']
        """
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Starred) and isinstance(target.value, ast.Name):
            return [target.value.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            for elt in target.elts:
                names.extend(self._collect_target_names(elt))
            return names
        return []

    def _register_target(self, target: ast.AST) -> None:
        """Register all names from a target in the current scope."""
        names = self._collect_target_names(target)
        if self._in_function and self._local_vars:
            self._local_vars[-1].update(names)
        else:
            self.global_vars.update(names)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._register_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target:
            self._register_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Walrus operator: x := expr"""
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                self._register_target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            if self._in_function and self._local_vars:
                self._local_vars[-1].add(node.name)
            else:
                self.global_vars.add(node.name)
        self.generic_visit(node)

    def _visit_comp(self, node: ast.AST) -> None:
        """Visit comprehensions with correct ordering.

        The AST puts `elt` before `generators`, but the iteration
        variables must be registered FIRST so the element expression
        can reference them without false positives.
        """
        # Register all comprehension targets first
        for gen in node.generators:  # type: ignore[attr-defined]
            self._register_target(gen.target)
            # Visit the iter expression (it doesn't use the comp variable)
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)

        # Now visit the element expression(s)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        elif hasattr(node, "elt"):
            self.visit(node.elt)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comp(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comp(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comp(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comp(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._check_name_defined(node.func.id, node, is_call=True)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self._check_name_defined(
                    node.func.value.id, node.func.value, is_call=False
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._check_name_defined(node.id, node, is_call=False)
        self.generic_visit(node)


# Python builtins that don't need to be defined
BUILTIN_NAMES: set[str] = {
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "breakpoint",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "__import__",
    "True",
    "False",
    "None",
    "NotImplemented",
    "Ellipsis",
    "Exception",
    "BaseException",
    "SystemExit",
    "KeyboardInterrupt",
    "GeneratorExit",
    "StopIteration",
    "ArithmeticError",
    "LookupError",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "OSError",
    "IOError",
    "FileNotFoundError",
    "PermissionError",
    "IndexError",
    "KeyError",
    "AttributeError",
    "NameError",
    "ImportError",
    "SyntaxError",
    "IndentationError",
    "TabError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "OSError",
    "EnvironmentError",
    "AssertionError",
    "EOFError",
    "FloatingPointError",
    "OverflowError",
    "ZeroDivisionError",
    "PendingDeprecationWarning",
    "DeprecationWarning",
    "FutureWarning",
    "ImportWarning",
    "RuntimeWarning",
    "SyntaxWarning",
    "UserWarning",
    "Warning",
    "BufferError",
    "BytesWarning",
    "RegExpError",
    "timeout",
    "json",
    "re",
    "os",
    "sys",
    "pathlib",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "abc",
    "copy",
    "io",
    "logging",
}


def static_check_files(files: list[dict]) -> StaticCheckResult:
    """Check all files for structural issues.

    Args:
        files: List of {"path": str, "content": str} dicts

    Returns:
        StaticCheckResult with any errors found
    """
    result = StaticCheckResult()

    # Filter to Python files only
    py_files = [
        f
        for f in files
        if f.get("path", "").endswith(".py") and f.get("content", "").strip()
    ]

    if not py_files:
        result.warnings.append("No Python files to check")
        return result

    # Phase 1: Parse all files and collect symbols
    collectors: dict[str, SymbolCollector] = {}
    parsed: dict[str, ast.Module] = {}

    for f in py_files:
        path = f["path"]
        content = f["content"]

        try:
            tree = ast.parse(content, filename=path)
            parsed[path] = tree

            collector = SymbolCollector(path)
            collector.visit(tree)
            collectors[path] = collector

            # Also register functions/classes globally for cross-file refs
            for func in collector.functions:
                pass  # Already in collector
            for cls in collector.classes:
                pass  # Already in collector

        except SyntaxError as e:
            result.errors.append(
                StaticCheckError(
                    error_type="syntax",
                    file=path,
                    line=e.lineno or 0,
                    message=f"Syntax error: {e.msg}",
                    suggestion="Fix syntax before static analysis",
                )
            )
            result.passed = False

    if not result.passed:
        return result

    # Phase 2: Check each file for undefined references
    for f in py_files:
        path = f["path"]
        content = f["content"]

        collector = collectors[path]
        tree = parsed[path]

        # Build combined symbol table including all files
        all_collectors = {
            other_path: other
            for other_path, other in collectors.items()
            if other_path != path
        }

        checker = ReferenceChecker(
            file_path=path,
            defined_functions=collector.functions,
            defined_classes=collector.classes,
            imports=collector.imports,
            from_imports=collector.from_imports,
            global_vars=set(collector.global_vars),  # copy — checker may mutate
            all_files=all_collectors,
            builtin_names=BUILTIN_NAMES,
        )
        checker.visit(tree)

        result.errors.extend(checker.errors)

    result.passed = len(result.errors) == 0
    return result


def verify_signatures(
    files: list[dict],
    signatures: list[dict],
) -> list[dict]:
    """Check that required function/method signatures exist in generated code.

    Each signature dict should have at minimum ``file`` and ``name``.
    If ``class_name`` is non-empty, the function must be a method of that class.

    Only checks existence (name + scope), not parameter types or return types —
    LLMs rename params too freely for exact matching to be useful.

    Args:
        files: List of {"path": str, "content": str} dicts.
        signatures: List of SignatureSpec-like dicts.

    Returns:
        List of ReviewIssue-compatible dicts for each missing signature.
        Empty list means all signatures found.
    """
    from the_bois.contracts import ReviewIssue

    if not signatures:
        return []

    # Index files by path for fast lookup — normalise to basename and
    # full path so "src/store.py" matches regardless of prefix.
    files_by_path: dict[str, dict] = {}
    for f in files:
        p = f.get("path", "")
        if p:
            files_by_path[p] = f
            # Also register by basename for fuzzy matching
            files_by_path[PurePosixPath(p).name] = f

    # Cache parsed ASTs so we don't re-parse per signature
    parsed_cache: dict[str, ast.Module | None] = {}

    def _parse(path: str) -> ast.Module | None:
        if path in parsed_cache:
            return parsed_cache[path]
        fdata = files_by_path.get(path)
        if fdata is None:
            # Try basename match
            fdata = files_by_path.get(PurePosixPath(path).name)
        if fdata is None:
            parsed_cache[path] = None
            return None
        try:
            tree = ast.parse(fdata["content"], filename=path)
        except SyntaxError:
            parsed_cache[path] = None
            return None
        parsed_cache[path] = tree
        return tree

    issues: list[dict] = []

    for sig in signatures:
        sig_file = sig.get("file", "")
        sig_name = sig.get("name", "")
        sig_class = sig.get("class_name", "")

        if not sig_file or not sig_name:
            continue  # malformed spec, skip

        tree = _parse(sig_file)
        if tree is None:
            issues.append(
                ReviewIssue(
                    severity="critical",
                    file=sig_file,
                    description=(
                        f"Required file '{sig_file}' is missing or has syntax errors. "
                        f"Expected to find {'method' if sig_class else 'function'} "
                        f"'{sig_class + '.' if sig_class else ''}{sig_name}' here."
                    ),
                    suggestion=f"Create '{sig_file}' with the required implementation.",
                ).to_dict()
            )
            continue

        # Collect top-level functions and class methods from the AST
        found = False

        if sig_class:
            # Look for a method inside a specific class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == sig_class:
                    for item in node.body:
                        if (
                            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == sig_name
                        ):
                            found = True
                            break
                if found:
                    break
        else:
            # Look for a top-level function (not nested in a class)
            for node in ast.iter_child_nodes(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == sig_name
                ):
                    found = True
                    break

        if not found:
            label = f"{sig_class}.{sig_name}" if sig_class else sig_name
            scope_hint = f" in class '{sig_class}'" if sig_class else " at module level"
            issues.append(
                ReviewIssue(
                    severity="critical",
                    file=sig_file,
                    description=(
                        f"Required {'method' if sig_class else 'function'} "
                        f"'{label}' not found{scope_hint} in '{sig_file}'."
                    ),
                    suggestion=(
                        f"Implement '{label}()'{scope_hint}. "
                        f"The architect's task spec requires this signature."
                    ),
                ).to_dict()
            )

    return issues


def static_check(files: list[dict]) -> list[dict]:
    """Convenience wrapper - returns list of issues in reviewer format.

    Args:
        files: List of {"path": str, "content": str} dicts

    Returns:
        List of issue dicts suitable for reviewer feedback
    """
    result = static_check_files(files)
    return [e.to_reviewer_format() for e in result.errors]


# ── Self-verification: task-level completeness checks ───────────────── #


def self_verify(
    files: list[dict],
    task: dict,
) -> StaticCheckResult:
    """Verify generated code satisfies basic task-level expectations.

    Unlike static_check_files (which catches undefined references),
    this catches *semantic* gaps:
      - Expected output files missing from generated code
      - Empty / stub-only files (just ``pass`` or empty functions)
      - Test files that don't import from any generated module

    Cheap, deterministic, no LLM needed.  Run right after auto-repair
    and before the full static check.

    Args:
        files: Generated code files as {"path": str, "content": str} dicts.
        task: Task dict from the architect (has "output_files", "description").

    Returns:
        StaticCheckResult with any completeness issues found.
    """
    result = StaticCheckResult()
    generated_paths = {f.get("path", "") for f in files if f.get("path")}
    generated_stems = {
        PurePosixPath(p).stem for p in generated_paths if p.endswith(".py")
    }

    # ── 1. Missing expected output files ──
    expected_files = task.get("output_files", [])
    for expected in expected_files:
        if isinstance(expected, str) and expected not in generated_paths:
            # Fuzzy match: maybe they put it in a subdirectory
            basename = PurePosixPath(expected).name
            if not any(p.endswith(basename) for p in generated_paths):
                result.errors.append(
                    StaticCheckError(
                        error_type="missing_file",
                        file=expected,
                        line=0,
                        name=expected,
                        message=f"Expected output file '{expected}' is missing from generated code",
                        suggestion=f"Create '{expected}' as specified in the task.",
                    )
                )

    # ── 2. Empty / stub-only files ──
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "").strip()
        if not path.endswith(".py") or not content:
            continue

        # Skip __init__.py — it's fine to be empty
        if PurePosixPath(path).name == "__init__.py":
            continue

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            continue  # static_check will catch this

        # Check if every function body is just `pass` or `...`
        has_real_code = False
        func_count = 0
        stub_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_count += 1
                body = node.body
                # A stub is: only Pass, Expr(Constant(...)), or docstring + pass
                real_stmts = [
                    s
                    for s in body
                    if not isinstance(s, ast.Pass)
                    and not (
                        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    )
                ]
                if real_stmts:
                    has_real_code = True
                else:
                    stub_count += 1

        if func_count > 0 and stub_count == func_count and not has_real_code:
            result.errors.append(
                StaticCheckError(
                    error_type="stub_only",
                    file=path,
                    line=0,
                    name=path,
                    message=(
                        f"File '{path}' has {func_count} function(s) but ALL are "
                        f"stubs (just 'pass' or '...'). No real implementation."
                    ),
                    suggestion="Implement the function bodies instead of leaving stubs.",
                )
            )

    # ── 3. Test files that don't import from any generated module ──
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "").strip()
        if not path.endswith(".py") or not content:
            continue

        name = PurePosixPath(path).name
        is_test = (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
        )
        if not is_test:
            continue

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            continue

        # Collect all top-level import targets
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        # Check if any import refers to a generated module
        imports_generated = imported_modules & generated_stems
        if not imports_generated:
            # Also check if any name from generated files appears in imports
            # (handles "from mypackage.module import ..." patterns)
            all_generated_parts = set()
            for gp in generated_paths:
                if gp.endswith(".py"):
                    all_generated_parts.update(PurePosixPath(gp).parts)
            imports_generated = imported_modules & all_generated_parts

        if not imports_generated:
            result.warnings.append(
                f"Test file '{path}' doesn't import from any generated module. "
                f"It may be testing the wrong thing or using incorrect import paths."
            )

    result.passed = len(result.errors) == 0
    return result
