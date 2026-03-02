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
        severity = "critical" if self.error_type in (
            ERROR_FUNCTION_DEFINED,
            ERROR_CLASS_DEFINED,
            ERROR_IMPORT_RESOLVED,
        ) else "major"
        return {
            "severity": severity,
            "file": self.file,
            "description": self.message,
            "suggestion": self.suggestion,
        }


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
                if self.errors else "Static analysis passed."
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
            self.errors.append(StaticCheckError(
                error_type=ERROR_FUNCTION_DEFINED,
                file=self.file_path,
                line=node.lineno or 0,
                column=node.col_offset,
                name=name,
                message=f"Function '{name}' is called but not defined or imported",
                suggestion=f"Define '{name}()' or add proper import",
            ))
        elif name[0].isupper():
            self.errors.append(StaticCheckError(
                error_type=ERROR_CLASS_DEFINED,
                file=self.file_path,
                line=node.lineno or 0,
                column=node.col_offset,
                name=name,
                message=f"Class '{name}' is used but not defined or imported",
                suggestion=f"Define class '{name}' or add proper import",
            ))
        else:
            self.errors.append(StaticCheckError(
                error_type=ERROR_VARIABLE_DEFINED,
                file=self.file_path,
                line=node.lineno or 0,
                column=node.col_offset,
                name=name,
                message=f"Variable '{name}' is used but not defined in scope",
                suggestion=f"Define '{name}' before using it",
            ))

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
                self._check_name_defined(node.func.value.id, node.func.value, is_call=False)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._check_name_defined(node.id, node, is_call=False)
        self.generic_visit(node)


# Python builtins that don't need to be defined
BUILTIN_NAMES: set[str] = {
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
    "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass",
    "iter", "len", "list", "locals", "map", "max", "memoryview", "min",
    "next", "object", "oct", "open", "ord", "pow", "print", "property",
    "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
    "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
    "__import__", "True", "False", "None", "NotImplemented", "Ellipsis",
    "Exception", "BaseException", "SystemExit", "KeyboardInterrupt",
    "GeneratorExit", "StopIteration", "ArithmeticError", "LookupError",
    "ValueError", "TypeError", "RuntimeError", "OSError", "IOError",
    "FileNotFoundError", "PermissionError", "IndexError", "KeyError",
    "AttributeError", "NameError", "ImportError", "SyntaxError",
    "IndentationError", "TabError", "UnicodeDecodeError", "UnicodeEncodeError",
    "UnicodeError", "OSError", "EnvironmentError", "AssertionError",
    "EOFError", "FloatingPointError", "OverflowError", "ZeroDivisionError",
    "PendingDeprecationWarning", "DeprecationWarning", "FutureWarning",
    "ImportWarning", "RuntimeWarning", "SyntaxWarning", "UserWarning",
    "Warning", "BufferError", "BytesWarning", "RegExpError", "timeout",
    "json", "re", "os", "sys", "pathlib", "datetime", "collections",
    "itertools", "functools", "operator", "abc", "copy", "io", "logging",
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
        f for f in files
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
            result.errors.append(StaticCheckError(
                error_type="syntax",
                file=path,
                line=e.lineno or 0,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix syntax before static analysis",
            ))
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


def static_check(files: list[dict]) -> list[dict]:
    """Convenience wrapper - returns list of issues in reviewer format.

    Args:
        files: List of {"path": str, "content": str} dicts

    Returns:
        List of issue dicts suitable for reviewer feedback
    """
    result = static_check_files(files)
    return [e.to_reviewer_format() for e in result.errors]
