from __future__ import annotations

from dataclasses import dataclass

# Logical role -> concrete model string. Call sites reference a role
# ("fast", "smart", ...) so you can repoint models in exactly one place.
MODEL_REGISTRY = {
    "fast": "claude-haiku-4-5-20251001",
    "smart": "claude-sonnet-5",
    "reasoning": "claude-opus-4-8",
}


@dataclass
class AgentConfig:
    model: str = "smart"       # a MODEL_REGISTRY key, or a raw model string
    max_tokens: int = 2048
    system: str | None = None
    max_iterations: int = 10   # hard cap so a runaway tool-use loop can't spin forever

    def resolved_model(self) -> str:
        return MODEL_REGISTRY.get(self.model, self.model)
