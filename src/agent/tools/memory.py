from __future__ import annotations

from pathlib import Path

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)


class ScopedFilesystemMemoryTool(BetaAbstractMemoryTool):
    """Subclass point for a custom storage backend (S3, a database, a
    per-user namespace, etc). This example still writes to local disk,
    but every command is now typed and validated by the SDK instead of
    hand-parsed from a raw dict -- Anthropic's docs recommend exactly
    this pattern over reimplementing the six commands from scratch.

    For the common case of "just use local disk," you don't need this
    class at all -- `from anthropic.tools import BetaLocalFilesystemMemoryTool`
    already does this out of the box. Use this subclass only when you
    need custom storage.
    """

    def __init__(self, base_path: str | Path = "./agent_memory"):
        super().__init__()
        self.root = Path(base_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        if not path.startswith("/memories"):
            raise ValueError(f"Path must start with /memories, got: {path}")
        rel = path[len("/memories"):].lstrip("/")
        real = (self.root / rel).resolve()
        if self.root not in real.parents and real != self.root:
            raise ValueError(f"Path traversal blocked: {path}")
        return real

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        target = self._resolve(command.path)
        if not target.exists():
            return f"The path {command.path} does not exist. Please provide a valid path."
        if target.is_dir():
            lines = [f"Directory listing for {command.path}:"]
            lines += [
                "/memories/" + str(p.relative_to(self.root))
                for p in sorted(target.rglob("*")) if p.is_file()
            ]
            return "\n".join(lines)
        return target.read_text()

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        target = self._resolve(command.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(command.file_text)
        return f"File created successfully at: {command.path}"

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        target = self._resolve(command.path)
        text = target.read_text()
        count = text.count(command.old_str)
        if count != 1:
            return f"Error: old_str must match exactly once (found {count}) in {command.path}"
        target.write_text(text.replace(command.old_str, command.new_str or "", 1))
        return "The memory file has been edited."

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        target = self._resolve(command.path)
        lines = target.read_text().splitlines(keepends=True)
        lines.insert(command.insert_line, command.insert_text)
        target.write_text("".join(lines))
        return f"The file {command.path} has been edited."

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        target = self._resolve(command.path)
        import shutil
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        return f"Successfully deleted {command.path}"

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        self._resolve(command.old_path).rename(self._resolve(command.new_path))
        return f"Successfully renamed {command.old_path} to {command.new_path}"
