"""Data-only skill discovery and registration for trusted core tools."""

from __future__ import annotations

import inspect
import logging
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Annotated, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from palace.skills.tools import (
    get_core_fact,
    search_archival_memory,
    update_core_fact,
    append_to_journal,
)


logger = logging.getLogger(__name__)
DEFAULT_SKILLS_DIR = Path("data/skills")


@dataclass(frozen=True)
class SkillSpec:
    """Declarative instructions loaded from a Markdown file."""

    name: str
    description: str
    instructions: str


def _annotation_to_schema(annotation: Any) -> dict[str, Any] | None:
    """Translate a Python type annotation into a JSON-schema fragment."""
    if annotation in {Any, inspect.Signature.empty}:
        return {}
    if annotation is None or annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Annotated:
        return _annotation_to_schema(arguments[0])
    if origin in {Union, types.UnionType}:
        non_null = [arg for arg in arguments if arg is not type(None)]
        if len(non_null) == 1:
            return _annotation_to_schema(non_null[0])
        variants = [_annotation_to_schema(arg) for arg in non_null]
        if any(variant is None for variant in variants):
            return None
        return {"anyOf": variants}
    if origin is Literal:
        values = list(arguments)
        if not values:
            return {}
        schema = _annotation_to_schema(type(values[0])) or {}
        schema["enum"] = values
        return schema

    if annotation is str or annotation is Path:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation in {list, tuple, set, frozenset} or origin in {
        list,
        tuple,
        set,
        frozenset,
        Sequence,
    }:
        item_annotation = arguments[0] if arguments else Any
        item_schema = _annotation_to_schema(item_annotation)
        if item_schema is None:
            return None
        return {"type": "array", "items": item_schema}
    if annotation is dict or origin in {dict, Mapping}:
        value_annotation = arguments[1] if len(arguments) == 2 else Any
        value_schema = _annotation_to_schema(value_annotation)
        if value_schema is None:
            return None
        return {"type": "object", "additionalProperties": value_schema}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]
        schema = _annotation_to_schema(type(values[0])) if values else {}
        if schema is None:
            return None
        schema["enum"] = values
        return schema

    # Application objects (for example dependency-injection hooks) are not
    # values that an LLM can provide as JSON.
    return None


def function_to_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an Ollama-compatible function schema from a Python callable."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = inspect.get_annotations(func, eval_str=False)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        annotation = hints.get(name, parameter.annotation)
        parameter_schema = _annotation_to_schema(annotation)
        if parameter_schema is None:
            if parameter.default is inspect.Signature.empty:
                raise TypeError(
                    f"Parameter {name!r} on {func.__name__} has an unsupported "
                    f"annotation: {annotation!r}"
                )
            continue

        properties[name] = parameter_schema
        if parameter.default is inspect.Signature.empty:
            required.append(name)

    docstring = inspect.getdoc(func) or ""
    description = docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "required": required,
                "properties": properties,
            },
        },
    }


class SkillRegistry:
    """Store trusted core tools and data-only Markdown skill definitions."""

    def __init__(self, skills_dir: Path = DEFAULT_SKILLS_DIR) -> None:
        self._functions: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, SkillSpec] = {}
        self.register_core_tools()
        self.load_markdown_skills(skills_dir)

    def _register_core_tool(self, func: Callable[..., Any]) -> None:
        """Register a trusted, application-owned core callable."""
        schema = function_to_schema(func)
        self._functions[func.__name__] = func
        self._schemas[func.__name__] = schema

    def register_core_tools(self) -> None:
        """Register MindPalace's built-in memory tools."""
        for func in (get_core_fact, update_core_fact, search_archival_memory, append_to_journal):
            self._register_core_tool(func)

    def load_markdown_skills(self, skills_dir: Path = DEFAULT_SKILLS_DIR) -> None:
        """Load declarative skill definitions from Markdown files only."""
        path = Path(skills_dir)
        if not path.is_dir():
            return

        for skill_path in sorted(path.glob("*.md")):
            try:
                skill = _parse_markdown_skill(skill_path)
            except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                logger.warning("Skipping invalid skill file %s: %s", skill_path, exc)
                continue

            self.skills[skill.name] = skill

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tool schemas in registration order."""
        return list(self._schemas.values())

    def get_function(self, name: str) -> Callable[..., Any] | None:
        """Return the registered callable named ``name``, if present."""
        return self._functions.get(name)

    def get_skill(self, name: str) -> SkillSpec | None:
        """Return a loaded declarative skill, if present."""
        return self.skills.get(name)

    def get_skill_prompt(self, skill_name: str) -> str:
        """Format a skill's identity and instructions for the model."""
        skill = self.get_skill(skill_name)
        if skill is None:
            raise KeyError(f"Unknown skill: {skill_name}")

        return (
            f"Active skill: {skill.name}\n"
            f"Description: {skill.description}\n\n"
            f"Skill instructions:\n{skill.instructions}"
        )


def _parse_markdown_skill(skill_path: Path) -> SkillSpec:
    """Parse one Markdown skill without importing or executing its contents."""
    text = skill_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("unterminated YAML frontmatter") from exc

    metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(metadata, dict):
        raise ValueError("YAML frontmatter must be a mapping")

    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("missing or invalid skill name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("missing or invalid skill description")

    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    return SkillSpec(
        name=name.strip(),
        description=description.strip(),
        instructions=instructions,
    )


__all__ = ["DEFAULT_SKILLS_DIR", "SkillRegistry", "SkillSpec", "function_to_schema"]
