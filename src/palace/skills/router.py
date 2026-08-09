"""Ollama-powered conversational router for MindPalace memory tools."""

from __future__ import annotations

from typing import Any

import ollama

from palace.skills.loader import SkillRegistry, SkillSpec


class MemoryAgent:
    """Maintain a conversation and route Ollama tool calls to memory tools."""

    max_iterations = 3

    def __init__(
        self,
        model: str = "llama3.1",
        system_prompt: str | None = None,
        registry: SkillRegistry | None = None,
        active_skill: str | None = None,
    ) -> None:
        self.model = model
        self.registry = registry if registry is not None else SkillRegistry()
        self.active_skill: SkillSpec | None = None
        self.messages: list[Any] = []
        self.history = self.messages
        self._skill_message: dict[str, str] | None = None
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        if active_skill is not None:
            self.set_active_skill(active_skill)

    def set_active_skill(self, skill_name: str | None) -> None:
        """Activate a loaded skill, or clear the current skill with ``None``."""
        if skill_name is None:
            self.active_skill = None
            self._remove_skill_message()
            return

        skill = self.registry.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Unknown skill: {skill_name}")

        self.active_skill = skill
        prompt = self.registry.get_skill_prompt(skill_name)
        if self._skill_message is None:
            self._skill_message = {"role": "system", "content": prompt}
            self.history.append(self._skill_message)
        else:
            self._skill_message["content"] = prompt

    def _remove_skill_message(self) -> None:
        """Remove the active skill prompt from conversation history."""
        if self._skill_message is None:
            return
        self.history[:] = [
            message for message in self.history if message is not self._skill_message
        ]
        self._skill_message = None

    def chat(self, user_message: str) -> str:
        """Respond to a user message, executing any requested memory tools."""
        self.history.append({"role": "user", "content": user_message})
        response = ollama.chat(
            model=self.model,
            messages=self.history,
            tools=self.registry.get_schemas(),
        )
        assistant_message = response.message
        self.history.append(assistant_message)

        iteration_count = 0
        while assistant_message.tool_calls and iteration_count < self.max_iterations:
            for tool_call in assistant_message.tool_calls:
                function = tool_call.function
                arguments = dict(function.arguments)
                policy_error = self._policy_error(function.name, arguments)
                if policy_error:
                    result = policy_error
                else:
                    tool = self.registry.get_function(function.name)
                    if tool is None:
                        raise ValueError(f"Unknown memory tool: {function.name}")
                    result = tool(**arguments)
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
                tools=self.registry.get_schemas(),
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

    def _policy_error(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Reject tool calls that exceed the active skill's permissions."""
        if self.active_skill is None:
            return None
        if tool_name not in self.active_skill.allowed_tools:
            return (
                f"Security Exception: Tool '{tool_name}' is not permitted by "
                "the active skill policy."
            )

        locker = arguments.get("locker_name")
        if locker is not None and locker not in self.active_skill.allowed_lockers:
            return (
                f"Security Exception: Access to locker '{locker}' is not permitted "
                "by the active skill policy."
            )
        return None


__all__ = ["MemoryAgent"]
