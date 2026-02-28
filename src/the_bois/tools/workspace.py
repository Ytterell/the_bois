"""Workspace file I/O — agents write their output here."""

from __future__ import annotations

from pathlib import Path


class Workspace:
    """Manages a directory where agents can read/write files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def write_file(self, filename: str, content: str) -> Path:
        """Write content to a file in the workspace."""
        filepath = self.path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        return filepath

    def read_file(self, filename: str) -> str:
        """Read a file from the workspace."""
        filepath = self.path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Not in workspace: {filename}")
        return filepath.read_text()

    def list_files(self) -> list[str]:
        """List all files in the workspace (relative paths)."""
        return sorted(
            str(p.relative_to(self.path))
            for p in self.path.rglob("*")
            if p.is_file() and p.name != ".gitkeep"
        )
