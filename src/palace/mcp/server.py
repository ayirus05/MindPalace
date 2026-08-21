"""Lightweight MCP server exposing MindPalace semantic search."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from palace.embedder import OllamaEmbedder
from palace.indexing.repository import LanceDBRepository
from palace.models.config import PalaceConfig
from palace.search.engine import SemanticSearchEngine


def _load_config() -> PalaceConfig:
    """Load the project config, falling back to cwd-rooted defaults."""
    config_path = Path.cwd() / "config.yaml"
    if config_path.is_file():
        return PalaceConfig.from_yaml(config_path)
    return PalaceConfig.default_for(Path.cwd())


@lru_cache(maxsize=1)
def _get_engine() -> SemanticSearchEngine:
    """Build the search engine lazily when the first tool call arrives."""
    config = _load_config()
    embedder = OllamaEmbedder(config.embedding)
    repository = LanceDBRepository(
        db_path=config.resolve(config.database.path),
        table_name=config.database.table_name,
    )
    return SemanticSearchEngine(config, embedder, repository)

mcp = MCPServer("MindPalace")


@mcp.tool()
def search_memory(
    query: str,
    domain: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[TextContent]:
    """Search semantic memory, optionally filtering by domain and date range."""
    outcome = _get_engine().search(
        query=query,
        domain=domain,
        date_from=date_from,
        date_to=date_to,
    )
    return [
        TextContent(type="text", text=result.model_dump_json())
        for result in outcome.results
    ]


if __name__ == "__main__":
    mcp.run()


__all__ = ["mcp", "search_memory"]
