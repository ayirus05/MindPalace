"""Thin, dependency-injected adapters for LLM memory tools.

The application or tool host injects both managers once during startup.  The
LLM-facing functions therefore expose only memory-related arguments and keep
storage construction out of the tool layer.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from palace.models.chunk import SearchResult
from palace.search.engine import SemanticSearchEngine
from palace.vault.manager import VaultManager

JOURNALS_DIR = Path("data/source_docs/journals")

_vault_manager: VaultManager | None = None
_search_engine: SemanticSearchEngine | None = None


def configure_memory_tools(
    *,
    vault_manager: VaultManager,
    search_engine: SemanticSearchEngine,
) -> None:
    """Inject the long-lived dependencies used by the LLM-facing functions."""
    global _vault_manager, _search_engine
    _vault_manager = vault_manager
    _search_engine = search_engine

# ---- Core Vault Tools ----

def get_core_fact(
    locker_name: str,
    field: str,
) -> Any:
    """Read an exact field from a Core Vault locker."""
    manager = _require_vault_manager()
    return manager.read_field(locker_name, field)


def update_core_fact(
    locker_name: str,
    field: str,
    value: Any,
) -> None:
    """Create or replace one JSON field in a local Core Vault locker."""
    manager = _require_vault_manager()
    manager.write_field(locker_name, field, value)
    return {"status": "success", "locker_name": locker_name, "field": field}

# ---- Archival Search Tools ----

def search_archival_memory(
    query: str,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Search archival memory and return matching chunks."""
    engine = _require_search_engine()
    results = engine.search(query=query, domain=domain).results
    return [_serialize_search_result(r) for r in results]

# ---- Journal Tools ----

def append_to_journal(date_str: str | None, entry: str) -> dict[str, Any]:
    """Append an entry as a Markdown bullet point in a daily journal file (e.g. YYYY-MM-DD)."""
    if not date_str:
        date_str = date.today().isoformat()
    filename = _journal_filename(date_str)
    path = JOURNALS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    prefix = ""
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as journal:
            journal.seek(-1, 2)
            if journal.read(1) != b"\n":
                prefix = "\n"

    with path.open("a", encoding="utf-8") as journal:
        journal.write(f"{prefix}- {entry}\n")

    return {"status": "success", "path": str(path)}

# ---- Helpers ----

def _serialize_search_result(
    result: SearchResult | dict[str, Any],
) -> dict[str, Any]:
    """Convert a typed search result to an LLM-tool-friendly dictionary."""
    if isinstance(result, dict):
        return dict(result)
    return result.model_dump(mode="json")

def _journal_filename(date_str: str) -> str:
    normalized = date_str.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        raise ValueError(f"Invalid journal date format: {date_str!r}")
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid journal date format: {date_str!r}") from exc
    return f"{normalized}.md"

def _require_vault_manager() -> VaultManager:
    if _vault_manager is None:
        raise RuntimeError(
            "Memory tools are not configured: inject a VaultManager with "
            "configure_memory_tools()"
        )
    return _vault_manager


def _require_search_engine() -> SemanticSearchEngine:
    if _search_engine is None:
        raise RuntimeError(
            "Memory tools are not configured: inject a search engine with "
            "configure_memory_tools()"
        )
    return _search_engine


__all__ = [
    "configure_memory_tools",
    "get_core_fact",
    "update_core_fact",
    "search_archival_memory",
    "append_to_journal",
]
