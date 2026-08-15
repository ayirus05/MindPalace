"""Thin, dependency-injected adapters for LLM memory tools.

The application or tool host injects both managers once during startup.  The
LLM-facing functions therefore expose only memory-related arguments and keep
storage construction out of the tool layer.
"""

from __future__ import annotations

from typing import Any, Protocol

from palace.models.chunk import SearchResult
from palace.search.engine import SearchOutcome
from palace.vault.manager import VaultManager


class ArchivalSearchEngine(Protocol):
    """Minimal search contract required by the archival-memory tool."""

    def search(
        self,
        query: str,
        domain: str | None = None,
    ) -> SearchOutcome:
        """Return ranked archival-memory results and diagnostics."""
        ...


_vault_manager: VaultManager | None = None
_search_engine: ArchivalSearchEngine | None = None


def configure_memory_tools(
    *,
    vault_manager: VaultManager,
    search_engine: ArchivalSearchEngine,
) -> None:
    """Inject the long-lived dependencies used by the LLM-facing functions."""
    global _vault_manager, _search_engine
    _vault_manager = vault_manager
    _search_engine = search_engine


def get_core_fact(
    locker_name: str,
    field: str,
    vault_manager: VaultManager | None = None,
) -> Any:
    """Read an exact field from a Core Vault locker.

    ``vault_manager`` is an injection hook for a host or test and should be
    omitted from the LLM tool schema.
    """
    manager = vault_manager or _require_vault_manager()
    return manager.read_field(locker_name, field)


def update_core_fact(
    locker_name: str,
    field: str,
    value: Any,
    vault_manager: VaultManager | None = None,
) -> None:
    """Create or update an exact field in a Core Vault locker.

    ``vault_manager`` is an injection hook for a host or test and should be
    omitted from the LLM tool schema.
    """
    manager = vault_manager or _require_vault_manager()
    manager.write_field(locker_name, field, value)


def search_archival_memory(
    query: str,
    domain: str | None = None,
    search_engine: ArchivalSearchEngine | None = None,
) -> list[dict[str, Any]]:
    """Search archival memory and return JSON-compatible result payloads.

    ``search_engine`` is an injection hook for a host or test and should be
    omitted from the LLM tool schema.
    """
    engine = search_engine or _require_search_engine()
    results = engine.search(query=query, domain=domain).results
    return [_serialize_search_result(result) for result in results]


def _serialize_search_result(
    result: SearchResult | dict[str, Any],
) -> dict[str, Any]:
    """Convert a typed search result to an LLM-tool-friendly dictionary."""
    if isinstance(result, dict):
        return dict(result)
    return result.model_dump(mode="json")


def _require_vault_manager() -> VaultManager:
    if _vault_manager is None:
        raise RuntimeError(
            "Memory tools are not configured: inject a VaultManager with "
            "configure_memory_tools()"
        )
    return _vault_manager


def _require_search_engine() -> ArchivalSearchEngine:
    if _search_engine is None:
        raise RuntimeError(
            "Memory tools are not configured: inject a search engine with "
            "configure_memory_tools()"
        )
    return _search_engine


__all__ = [
    "ArchivalSearchEngine",
    "configure_memory_tools",
    "get_core_fact",
    "update_core_fact",
    "search_archival_memory",
]
