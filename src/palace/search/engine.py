"""Semantic search engine — the public Python API.

This is the class a future MCP server will wrap.  The interface is plain
Python types (no MCP-specific imports), so wrapping it in an MCP tool is a
thin adapter layer rather than a reimplementation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from palace.embedder import OllamaEmbedder
from palace.indexing.repository import LanceDBRepository
from palace.models.chunk import DocumentDomain, SearchResult
from palace.models.config import PalaceConfig
from palace.models.search import SearchFilter
from palace.search.hybrid import HybridSearcher
from palace.search.ranker import SearchRanker


logger = logging.getLogger("palace.search.engine")


@dataclass
class SearchOutcome:
    """Bundled return value carrying results + observability metadata."""

    results: list[SearchResult]
    duration_ms: float
    candidate_count: int
    degraded_mode: bool = False


class SemanticSearchEngine:
    """High-level semantic search over the local LanceDB index.

    Responsibilities:
      * Coerce public filter arguments into repository-ready types.
      * Delegate retrieval and ranking to :class:`HybridSearcher`.
      * Return clean :class:`SearchResult` objects.

    The class holds no per-call state, so it's safe to reuse across calls.
    """

    def __init__(
        self,
        config: PalaceConfig,
        embedder: OllamaEmbedder,
        repository: LanceDBRepository,
        ranker: SearchRanker | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._repository = repository
        self._ranker = ranker or SearchRanker(config.search)
        self._hybrid_searcher = HybridSearcher(
            repository=self._repository,
            embedding_model=self._embedder,
            ranker=self._ranker,
        )

    def search(
        self,
        query: str,
        domain: str | DocumentDomain | None = None,
        date_from: date | str | None = None,
        date_to: date | str | None = None,
        tags: list[str] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> SearchOutcome:
        """Search the index and return results with timing diagnostics."""
        t0 = time.monotonic()

        search_filter = SearchFilter.model_validate(
            {
                "domain": domain,
                "date_from": date_from,
                "date_to": date_to,
                "tags": tags,
                "metadata_filters": metadata_filters,
                "top_k": top_k,
            }
        )
        filters = search_filter.model_dump(exclude_none=True)
        hybrid_outcome = self._hybrid_searcher.search(query, **filters)
        if isinstance(hybrid_outcome, tuple):
            results, degraded_mode = hybrid_outcome
        else:
            # Preserve compatibility with injected search doubles.
            results = hybrid_outcome
            degraded_mode = False
        if minimum_score is not None:
            results = [result for result in results if result.score >= minimum_score]
        if search_filter.top_k is not None:
            results = results[:search_filter.top_k]

        candidate_count = len(results)
        duration_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Search '%s' -> %d results in %.1fms (candidates=%d)",
            query[:60], len(results), duration_ms, candidate_count,
        )
        return SearchOutcome(
            results=results,
            duration_ms=duration_ms,
            candidate_count=candidate_count,
            degraded_mode=degraded_mode,
        )


__all__ = ["SemanticSearchEngine", "SearchOutcome"]
