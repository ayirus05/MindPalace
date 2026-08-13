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


def test_agent_blocks_disallowed_locker_and_reports_tool_error(
    tmp_path: Any, monkeypatch: Any
) -> None:
    (tmp_path / "restricted.md").write_text(
        """---
name: restricted
description: Read only the public profile.
allowed_tools: [get_core_fact]
allowed_lockers: [public]
---
Only read approved facts.
""",
        encoding="utf-8",
    )
    registry = SkillRegistry(skills_dir=tmp_path)
    executed = False

    def forbidden_call(**kwargs: Any) -> str:
        nonlocal executed
        executed = True
        return "secret"

    registry._functions["get_core_fact"] = forbidden_call
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
            SimpleNamespace(message=SimpleNamespace(content="Denied", tool_calls=[])),
        ]
    )
    monkeypatch.setattr(
        "palace.skills.router.ollama.chat", lambda **kwargs: next(responses)
    )

    agent = MemoryAgent(registry=registry, active_skills=["restricted"])
    assert agent.chat("Read my private email") == "Denied"
    assert executed is False
    assert agent.history[-2]["content"] == (
        "Security Exception: Access to locker 'private' is not permitted by "
        "the active skill policy."
    )
    assert "Allowed vault lockers: public" in agent.history[0]["content"]


def test_agent_unions_and_changes_active_skill_permissions(tmp_path: Any) -> None:
    skill_policies = (
        ("public-skill", "public", "get_core_fact"),
        ("work-skill", "work", "update_core_fact"),
    )
    for name, locker, tool in skill_policies:
        (tmp_path / f"{name}.md").write_text(
            f"""---
name: {name}
description: Access {locker} data.
allowed_tools: [{tool}]
allowed_lockers: [{locker}]
---
Use only the {locker} locker.
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
    assert agent._policy_error("get_core_fact", {"locker_name": "work"}) is None
    assert agent._policy_error("update_core_fact", {"locker_name": "public"}) is None
    assert agent._policy_error("search_archival_memory", {}) is not None

    initial_skill_messages = [
        message
        for message in agent.history
        if isinstance(message, dict) and "Active skill:" in message.get("content", "")
    ]
    assert len(initial_skill_messages) == 2

    agent.set_active_skills(["work-skill"])
    assert [skill.name for skill in agent.active_skills] == ["work-skill"]
    assert agent._policy_error("get_core_fact", {"locker_name": "work"}) is not None
    assert agent._policy_error("update_core_fact", {"locker_name": "work"}) is None
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
    assert agent._policy_error("get_core_fact", {"locker_name": "private"}) is None
    assert skill_messages[0] not in agent.history
