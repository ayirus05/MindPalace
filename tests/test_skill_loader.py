"""Tests for data-only skill discovery and schema generation."""

from __future__ import annotations

from pathlib import Path

from palace.skills.loader import SkillRegistry, function_to_schema


def example_tool(name: str, count: int = 1) -> list[str]:
    """Repeat a name."""
    return [name] * count


def test_function_to_schema_uses_signature_and_docstring() -> None:
    schema = function_to_schema(example_tool)["function"]

    assert schema["name"] == "example_tool"
    assert schema["description"] == "Repeat a name."
    assert schema["parameters"] == {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }


def test_registry_registers_core_tools(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_dir=tmp_path)

    names = {schema["function"]["name"] for schema in registry.get_schemas()}
    assert names == {
        "get_core_fact",
        "update_core_fact",
        "search_archival_memory",
    }
    get_schema = next(
        schema["function"]
        for schema in registry.get_schemas()
        if schema["function"]["name"] == "get_core_fact"
    )
    assert set(get_schema["parameters"]["properties"]) == {"locker_name", "field"}
    assert registry.get_function("get_core_fact") is not None


def test_registry_loads_markdown_skill(tmp_path: Path) -> None:
    (tmp_path / "journal.md").write_text(
        """---
name: journal-review
description: Review journal entries safely.
allowed_tools:
  - get_core_fact
  - search_archival_memory
allowed_lockers: [profile]
---
# Instructions

Summarize the user's recent entries.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(skills_dir=tmp_path)

    skill = registry.get_skill("journal-review")
    assert skill is not None
    assert skill.name == "journal-review"
    assert skill.description == "Review journal entries safely."
    assert skill.instructions.startswith("# Instructions")
    prompt = registry.get_skill_prompt("journal-review")
    assert "Active skill: journal-review" in prompt
    assert "Description: Review journal entries safely." in prompt
    assert "Summarize the user's recent entries." in prompt
    assert "Allowed vault lockers" not in prompt


def test_registry_never_executes_python_files(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    (tmp_path / "malicious.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    SkillRegistry(skills_dir=tmp_path)

    assert not marker.exists()


def test_invalid_markdown_skills_are_skipped(tmp_path: Path, caplog) -> None:
    (tmp_path / "invalid-yaml.md").write_text(
        "---\nname: [invalid\n---\nBody\n", encoding="utf-8"
    )
    (tmp_path / "missing-description.md").write_text(
        "---\nname: incomplete\n---\nBody\n", encoding="utf-8"
    )

    registry = SkillRegistry(skills_dir=tmp_path)

    assert registry.skills == {}
    assert caplog.text.count("Skipping invalid skill file") == 2
