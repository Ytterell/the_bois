"""Shared utilities — mostly dealing with local models being sloppy."""

from __future__ import annotations

import json
import re


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response, handling the usual nonsense.

    Local models love wrapping JSON in markdown fences, adding commentary
    before/after the JSON, or just generally being weird about it.
    """
    # Best case: it's already valid JSON
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from ```json ... ``` blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Brute-force: find the outermost { ... }
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that local models love to add.

    Handles ```python, ```json, plain ```, etc.
    """
    text = text.strip()
    # Remove opening fence: ```lang\n or ```\n
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_delimited_files(text: str) -> list[dict] | None:
    """Parse the delimiter-based code format used by the Coder agent.

    Expected format:
        ---FILE: path/to/file.ext---
        <code>
        ---END---

    Returns a list of {"path": ..., "content": ...} dicts, or None.
    """
    files: list[dict] = []
    # Split on file markers
    parts = re.split(r"---FILE:\s*(.+?)\s*---", text)
    # parts[0] is text before first marker, then alternating (path, content)
    for i in range(1, len(parts) - 1, 2):
        path = parts[i].strip()
        content = parts[i + 1]
        # Remove trailing ---END--- marker
        content = re.sub(r"---END---.*", "", content, flags=re.DOTALL).strip()
        content = strip_markdown_fences(content)
        if path and content:
            files.append({"path": path, "content": content})
    return files if files else None


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for display, preserving meaning."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def estimate_tokens(text: str) -> int:
    """Rough token estimate — content-aware.

    Code has more tokens per character than prose (~3 chars/token vs ~4).
    Detects code-heavy content by checking for common indicators.
    """
    code_indicators = ("def ", "class ", "import ", "return ", "    ", "{", "}", "=>")
    code_ratio = sum(1 for ind in code_indicators if ind in text) / len(code_indicators)
    chars_per_token = 3 if code_ratio > 0.3 else 4
    return len(text) // chars_per_token + 1
