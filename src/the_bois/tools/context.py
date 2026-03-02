"""Context optimization utilities — smarter context management for agents.

This module provides utilities for:
- File chunking: extracting relevant sections from large files
- Context profiles: defining what each agent needs in context
- Smart windowing: managing conversation history within token budgets
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from the_bois.utils import estimate_tokens


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
    lines = content.split('\n')
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
    definitions: list[tuple[int, int, str, str]] = []  # (start_line, end_line, type, name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                definitions.append((
                    node.lineno,
                    node.end_lineno or node.lineno,
                    type(node).__name__,
                    node.name,
                ))
    
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
            content='\n'.join(lines[:max_lines]),
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
        kept_lines_list.extend(lines[start-1:end])
    
    chunked_content = '\n'.join(kept_lines_list)
    
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
    identifiers = set(re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', text.lower()))
    
    # Remove common stopwords
    stopwords = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'has',
        'are', 'was', 'were', 'been', 'being', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'function', 'class', 'method', 'return', 'param', 'args', 'kwargs',
        'import', 'from', 'define', 'implement', 'create', 'add', 'make',
        'use', 'using', 'file', 'files', 'module', 'package', 'data',
        'test', 'tests', 'write', 'read', 'save', 'load', 'handle',
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
        if not path.endswith('.py'):
            result.append(f)
            continue
            
        chunked = extract_relevant_code(content, path, task_description, max_file_lines)
        
        if chunked.is_chunked:
            # Track that this was chunked for logging
            result.append({
                "path": path,
                "content": chunked.content,
                "_chunked": True,
                "_original_lines": chunked.original_lines,
                "_kept_lines": chunked.kept_lines,
            })
        else:
            result.append(f)
    
    return result


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
