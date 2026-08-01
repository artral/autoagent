from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class Tool(ABC):
    """Base class every tool (memory, weather, custom...) implements.

    Subclasses expose the JSON schema Claude sees via `schema`, and
    execute their side effect via `run`. This is what lets the agent
    core stay generic -- it never needs an if/elif per tool name.
    """

    name: str

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the Anthropic API."""

    @abstractmethod
    async def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """Execute the tool and return its result."""


class ToolRegistry:
    """Holds a set of tools and dispatches tool_use blocks by name.

    Add a new tool by instantiating it and passing it in here --
    nothing else in the codebase needs to change.
    """

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"No handler registered for tool: {name}", is_error=True)
        try:
            return await tool.run(tool_input)
        except Exception as exc:  # noqa: BLE001 -- surface failures to Claude, don't crash the loop
            return ToolResult(f"Tool '{name}' raised: {exc}", is_error=True)
