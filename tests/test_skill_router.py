"""Tests for routing model tool calls through a skill registry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ollama import Message

from palace.skills.loader import SkillRegistry
from palace.skills.router import MemoryAgent


class StubRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Remember a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def get_schemas(self) -> list[dict[str, Any]]:
        return self.schemas

    def get_function(self, name: str) -> Any:
        if name != "remember":
            return None

        def remember(**kwargs: Any) -> str:
            self.calls.append(kwargs)
            return "stored"

        return remember


class FailingRegistry(StubRegistry):
    def get_function(self, name: str) -> Any:
        if name != "remember":
            return None

        def remember(**kwargs: Any) -> str:
            raise RuntimeError("storage unavailable")

        return remember


def test_agent_uses_injected_registry(monkeypatch: Any) -> None:
    registry = StubRegistry()
    responses = iter(
        [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="remember", arguments={"value": "blue"}
                            )
                        )
                    ],
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(content="Done", tool_calls=[])
            ),
        ]
    )
    ollama_calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> Any:
        ollama_calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("palace.skills.router.ollama.chat", fake_chat)

    agent = MemoryAgent(registry=registry)  # type: ignore[arg-type]
    assert agent.chat("Remember blue") == "Done"
    assert registry.calls == [{"value": "blue"}]
    assert ollama_calls[0]["tools"] is registry.schemas
    assert ollama_calls[1]["tools"] is registry.schemas
    assistant_history = next(
        message
        for message in ollama_calls[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    Message.model_validate(assistant_history)


def test_agent_returns_tool_execution_errors_to_the_model() -> None:
    registry = FailingRegistry()
    responses = iter(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="remember", arguments={"value": "blue"}
                        )
                    )
                ],
            ),
            SimpleNamespace(content="Could not store it", tool_calls=[]),
        ]
    )
    provider = SimpleNamespace(chat=lambda **kwargs: next(responses))

    agent = MemoryAgent(
        registry=registry,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    assert agent.chat("Remember blue") == "Could not store it"
    assert agent.history[-2] == {
        "role": "tool",
        "tool_name": "remember",
        "content": "Tool execution failed: storage unavailable",
    }


def test_agent_executes_requested_tool_without_policy_gate(
    tmp_path: Any, monkeypatch: Any
) -> None:
    (tmp_path / "restricted.md").write_text(
        """---
name: restricted
description: Read whatever the caller asks for.
---
Only read approved facts.
""",
        encoding="utf-8",
    )
    registry = SkillRegistry(skills_dir=tmp_path)
    executed = False

    def call_tool(**kwargs: Any) -> str:
        nonlocal executed
        executed = True
        return f"secret:{kwargs.get('locker_name')}:{kwargs.get('field')}"

    registry._functions["get_core_fact"] = call_tool
    responses = iter(
        [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="get_core_fact",
                                arguments={"locker_name": "private", "field": "email"},
                            )
                        )
                    ],
                )
            ),
            SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=[])),
        ]
    )
    monkeypatch.setattr(
        "palace.skills.router.ollama.chat", lambda **kwargs: next(responses)
    )

    agent = MemoryAgent(registry=registry, active_skills=["restricted"])
    assert agent.chat("Read my private email") == "Done"
    assert executed is True
    assert agent.history[-2]["content"] == "secret:private:email"
    assert "Active skill: restricted" in agent.history[0]["content"]


def test_agent_tracks_active_skills_without_security_policy(tmp_path: Any) -> None:
    for name, description in (
        ("public-skill", "Access public data."),
        ("work-skill", "Access work data."),
    ):
        (tmp_path / f"{name}.md").write_text(
            f"""---
name: {name}
description: {description}
---
Use the {name} instructions.
""",
            encoding="utf-8",
        )

    agent = MemoryAgent(
        registry=SkillRegistry(skills_dir=tmp_path),
        active_skills=["public-skill", "work-skill"],
    )
    assert [skill.name for skill in agent.active_skills] == [
        "public-skill",
        "work-skill",
    ]

    initial_skill_messages = [
        message
        for message in agent.history
        if isinstance(message, dict) and "Active skill:" in message.get("content", "")
    ]
    assert len(initial_skill_messages) == 2

    agent.set_active_skills(["work-skill"])
    assert [skill.name for skill in agent.active_skills] == ["work-skill"]
    skill_messages = [
        message
        for message in agent.history
        if isinstance(message, dict) and "Active skill:" in message.get("content", "")
    ]
    assert len(skill_messages) == 1
    assert "Active skill: work-skill" in skill_messages[0]["content"]
    assert "Current active skills: work-skill" in agent.history[-2]["content"]

    agent.set_active_skills(None)
    assert agent.active_skills == []
    assert skill_messages[0] not in agent.history
