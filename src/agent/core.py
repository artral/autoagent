from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from anthropic import AsyncAnthropic, NOT_GIVEN

from .config import AgentConfig

MEMORY_BETA = "context-management-2025-06-27"


@dataclass
class AgentResult:
    text: str
    messages: list


class Agent:
    """Thin wrapper around the SDK's own `tool_runner`.

    tool_runner already implements the tool-use loop (call, execute,
    send tool_result, repeat until stop_reason != "tool_use") -- so
    this class no longer hand-rolls it. Its job is just to apply your
    AgentConfig (model, system prompt, iteration cap) and expose hooks
    for streaming text / logging tool calls back to the caller.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list,  # @beta_tool-decorated functions and/or BetaAbstractMemoryTool instances
        client: AsyncAnthropic | None = None,
    ):
        self.config = config
        self.tools = tools
        self.client = client or AsyncAnthropic()

    async def run(
        self,
        user_message: str,
        on_text: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ) -> AgentResult:
        messages: list = [{"role": "user", "content": user_message}]

        runner = self.client.beta.messages.tool_runner(
            betas=[MEMORY_BETA],  # only needed while any memory tool is in `self.tools`
            model=self.config.resolved_model(),
            max_tokens=self.config.max_tokens,
            system=self.config.system or NOT_GIVEN,
            tools=self.tools,
            messages=messages,
            max_iterations=self.config.max_iterations,
        )

        final_text = ""
        async for message in runner:
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    final_text = block.text
                    if on_text:
                        on_text(block.text)
                elif block.type == "tool_use" and on_tool_call:
                    on_tool_call(block.name, block.input)

        return AgentResult(text=final_text, messages=messages)
