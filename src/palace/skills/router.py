"""Provider-backed conversational router for MindPalace memory tools."""

from __future__ import annotations

from typing import Any

import ollama

from palace.llm.provider import BaseLLMProvider, OllamaProvider
from palace.skills.loader import SkillRegistry, SkillSpec


class MemoryAgent:
    """Maintain a conversation and route tool calls to memory tools."""

    max_iterations = 3

    def __init__(
        self,
        model: str = "llama3.1",
        system_prompt: str | None = None,
        registry: SkillRegistry | None = None,
        active_skills: list[str] | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self.model = model
        self.provider = provider or OllamaProvider(model_name=model)
        self.registry = registry if registry is not None else SkillRegistry()
        self.active_skills: list[SkillSpec] = []
        self.messages: list[Any] = []
        self.history = self.messages
        self._skill_messages: list[dict[str, str]] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        if active_skills:
            self.active_skills.extend(self._resolve_skills(active_skills))
            self._append_skill_prompts()

    def set_active_skills(self, skill_names: list[str] | None) -> None:
        """Replace the active skills and notify the model of the policy change."""
        skills = self._resolve_skills(skill_names or [])
        self._remove_skill_prompts()
        self.active_skills = skills

        names = ", ".join(skill.name for skill in skills) or "none"
        self.history.append(
            {
                "role": "system",
                "content": f"System: Active skills changed. Current active skills: {names}.",
            }
        )
        self._append_skill_prompts()

    def _resolve_skills(self, skill_names: list[str]) -> list[SkillSpec]:
        """Resolve all skill names before changing the active policy."""
        skills: list[SkillSpec] = []
        for skill_name in skill_names:
            skill = self.registry.get_skill(skill_name)
            if skill is None:
                raise ValueError(f"Unknown skill: {skill_name}")
            skills.append(skill)
        return skills

    def _append_skill_prompts(self) -> None:
        """Append each active skill's prompt as a separate system message."""
        for skill in self.active_skills:
            message = {
                "role": "system",
                "content": self.registry.get_skill_prompt(skill.name),
            }
            self._skill_messages.append(message)
            self.history.append(message)

    def _remove_skill_prompts(self) -> None:
        """Remove prompts belonging to the previously active skills."""
        self.history[:] = [
            message
            for message in self.history
            if all(message is not skill_message for skill_message in self._skill_messages)
        ]
        self._skill_messages.clear()

    def chat(self, user_message: str) -> str:
        """Respond to a user message, executing any requested memory tools."""
        self.history.append({"role": "user", "content": user_message})
        response = self.provider.chat(
            messages=self.history,
            tools=self.registry.get_schemas(),
        )
        assistant_message = response
        print(f"[MemoryAgent] level=0 response: {assistant_message.content!r}")
        if assistant_message.tool_calls:
            print(f"[MemoryAgent] level=0 tool_calls: {assistant_message.tool_calls!r}")
        self.history.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls,
        })

        iteration_count = 0
        while assistant_message.tool_calls and iteration_count < self.max_iterations:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.get("name")
                if not function_name:
                    continue
                arguments = dict(tool_call.get("arguments") or {})
                policy_error = self._policy_error(function_name, arguments)
                if policy_error:
                    result = policy_error
                else:
                    tool = self.registry.get_function(function_name)
                    if tool is None:
                        raise ValueError(f"Unknown memory tool: {function_name}")
                    result = tool(**arguments)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_name": function_name,
                        "content": str(result),
                    }
                )

            response = self.provider.chat(
                messages=self.history,
                tools=self.registry.get_schemas(),
            )
            assistant_message = response
            print(f"[MemoryAgent] level={iteration_count + 1} response: {assistant_message.content!r}")
            if assistant_message.tool_calls:
                print(f"[MemoryAgent] level={iteration_count + 1} tool_calls: {assistant_message.tool_calls!r}")
            self.history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls,
            })
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
            response = self.provider.chat(
                messages=self.history,
            )
            assistant_message = response
            print(f"[MemoryAgent] level=max_iterations response: {assistant_message.content!r}")
            if assistant_message.tool_calls:
                print(f"[MemoryAgent] level=max_iterations tool_calls: {assistant_message.tool_calls!r}")
            self.history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls,
            })

        return assistant_message.content or ""

    def _policy_error(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Reject tool calls that exceed the active skill's permissions."""
        if not self.active_skills:
            return None

        allowed_tools = {
            tool
            for skill in self.active_skills
            for tool in skill.allowed_tools
        }
        if tool_name not in allowed_tools:
            return (
                f"Security Exception: Tool '{tool_name}' is not permitted by "
                "the active skill policy."
            )

        locker = arguments.get("locker_name")
        allowed_lockers = {
            locker_name
            for skill in self.active_skills
            for locker_name in skill.allowed_lockers
        }
        if locker is not None and locker not in allowed_lockers:
            return (
                f"Security Exception: Access to locker '{locker}' is not permitted "
                "by the active skill policy."
            )
        return None


__all__ = ["MemoryAgent"]
