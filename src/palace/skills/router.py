"""Ollama-powered conversational router for MindPalace memory tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ollama

from palace.skills.memory_tools import (
    get_core_fact,
    search_archival_memory,
    update_core_fact,
)


MEMORY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_core_fact",
            "description": "Read an exact field from a Core Vault locker.",
            "parameters": {
                "type": "object",
                "required": ["locker_name", "field"],
                "properties": {
                    "locker_name": {
                        "type": "string",
                        "description": "Name of the Core Vault locker.",
                    },
                    "field": {
                        "type": "string",
                        "description": "Exact field key to read.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_core_fact",
            "description": "Create or update an exact Core Vault field.",
            "parameters": {
                "type": "object",
                "required": ["locker_name", "field", "value"],
                "properties": {
                    "locker_name": {
                        "type": "string",
                        "description": "Name of the Core Vault locker.",
                    },
                    "field": {
                        "type": "string",
                        "description": "Exact field key to create or update.",
                    },
                    "value": {
                        "description": "JSON-compatible value to store.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_archival_memory",
            "description": "Semantically search archival MindPalace memory.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language memory search query.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional memory domain filter.",
                    },
                },
            },
        },
    },
]


MEMORY_TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_core_fact": get_core_fact,
    "update_core_fact": update_core_fact,
    "search_archival_memory": search_archival_memory,
}


class MemoryAgent:
    """Maintain a conversation and route Ollama tool calls to memory tools."""

    max_iterations = 3

    def __init__(
        self,
        model: str = "llama3.1",
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.messages: list[Any] = []
        self.history = self.messages
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Respond to a user message, executing any requested memory tools."""
        self.history.append({"role": "user", "content": user_message})
        response = ollama.chat(
            model=self.model,
            messages=self.history,
            tools=MEMORY_TOOL_SCHEMAS,
        )
        assistant_message = response.message
        self.history.append(assistant_message)

        iteration_count = 0
        while assistant_message.tool_calls and iteration_count < self.max_iterations:
            for tool_call in assistant_message.tool_calls:
                function = tool_call.function
                tool = MEMORY_TOOL_FUNCTIONS.get(function.name)
                if tool is None:
                    raise ValueError(f"Unknown memory tool: {function.name}")

                result = tool(**dict(function.arguments))
                self.history.append(
                    {
                        "role": "tool",
                        "tool_name": function.name,
                        "content": str(result),
                    }
                )

            response = ollama.chat(
                model=self.model,
                messages=self.history,
                tools=MEMORY_TOOL_SCHEMAS,
            )
            assistant_message = response.message
            self.history.append(assistant_message)
            iteration_count += 1

        if assistant_message.tool_calls and iteration_count >= self.max_iterations:
            self.history.append(
                {
                    "role": "system",
                    "content": (
                        "System: Maximum tool iterations reached. Please provide a "
                        "final answer based on the context gathered so far."
                    ),
                }
            )
            response = ollama.chat(
                model=self.model,
                messages=self.history,
            )
            assistant_message = response.message
            self.history.append(assistant_message)

        return assistant_message.content or ""


__all__ = ["MEMORY_TOOL_SCHEMAS", "MEMORY_TOOL_FUNCTIONS", "MemoryAgent"]
