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

from palace.embeddings.manager import Embedder
from palace.indexing.repository import ChunkRepository
from palace.models.chunk import DocumentDomain, RawHit, SearchResult
from palace.models.config import PalaceConfig
from palace.search.ranker import SearchRanker


logger = logging.getLogger("palace.search.engine")


@dataclass
class SearchOutcome:
    """Bundled return value carrying results + observability metadata."""

    results: list[SearchResult]
    duration_ms: float
    candidate_count: int


class SemanticSearchEngine:
    """High-level semantic search over the local LanceDB index.

    Responsibilities:
      * Embed the query via the configured :class:`Embedder`.
      * Pre-filter on metadata (domain / date / tags) where possible.
      * Delegate ranking to :class:`SearchRanker`.
      * Return clean :class:`SearchResult` objects.

    The class holds no per-call state, so it's safe to reuse across calls.
    """

    def __init__(
        self,
        config: PalaceConfig,
        embedder: Embedder,
        repository: ChunkRepository,
        ranker: SearchRanker | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._repository = repository
        self._ranker = ranker or SearchRanker(config.search)

    def semantic_search(
        self,
        query: str,
        domain: str | DocumentDomain | None = None,
        date_from: date | str | None = None,
        date_to: date | str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        """Search the index, returning ranked results.

        Args:
            query: Natural language query.
            domain: Restrict to a single domain (string or enum).
            date_from: Earliest entry date (inclusive).  Accepts ``date`` or ISO string.
            date_to: Latest entry date (inclusive).
            tags: Only return chunks containing all these tags.
            top_k: Override the default number of results.
            minimum_score: Override the default minimum similarity.

        Returns:
            A list of :class:`SearchResult`, best first.
        """
        outcome = self.search_with_diagnostics(
            query=query,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            tags=tags,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        return outcome.results

    def search_with_diagnostics(
        self,
        query: str,
        domain: str | DocumentDomain | None = None,
        date_from: date | str | None = None,
        date_to: date | str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> SearchOutcome:
        """Like :meth:`semantic_search` but also returns timing + candidate count."""
        t0 = time.monotonic()

        # Coerce types.
        domain_enum = _coerce_domain(domain)
        d_from = _coerce_date(date_from)
        d_to = _coerce_date(date_to)

        # Embed query.
        query_vector = self._embedder.embed_one(query)

        # Vector search with pre-filtering.
        prefilter_k = self._config.search.prefilter_top_k
        raw = self._repository.search(
            query_vector=query_vector,
            top_k=prefilter_k,
            domain=domain_enum,
            date_from=d_from,
            date_to=d_to,
            tags=tags,
        )
        candidate_count = len(raw)
        logger.debug("Vector search returned %d candidates", candidate_count)

        # Convert to RawHit for the ranker.
        hits = [_row_to_hit(r) for r in raw]

        # Apply overrides on a copy of the ranker config.
        ranker = self._ranker
        if top_k is not None or minimum_score is not None:
            new_search_cfg = self._config.search.model_copy(
                update={
                    "default_top_k": top_k if top_k is not None else self._config.search.default_top_k,
                    "minimum_score": minimum_score if minimum_score is not None else self._config.search.minimum_score,
                }
            )
            ranker = SearchRanker(new_search_cfg)

        results = ranker.rank(hits)
        duration_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Search '%s' -> %d results in %.1fms (candidates=%d)",
            query[:60], len(results), duration_ms, candidate_count,
        )
        return SearchOutcome(
            results=results,
            duration_ms=duration_ms,
            candidate_count=candidate_count,
        )


# ---- helpers --------------------------------------------------------------


def _coerce_domain(value: str | DocumentDomain | None) -> DocumentDomain | None:
    if value is None:
        return None
    if isinstance(value, DocumentDomain):
        return value
    return DocumentDomain.from_string(value)


def _coerce_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse date filter: %r", value)
        return None


def _row_to_hit(row: dict[str, Any]) -> RawHit:
    """Convert a repository row dict to a :class:`RawHit`."""
    d = row.get("date")
    parsed_date: date | None = None
    if d:
        try:
            parsed_date = date.fromisoformat(d)
        except ValueError:
            parsed_date = None
    domain_value = row.get("domain")
    domain = (
        domain_value
        if isinstance(domain_value, DocumentDomain)
        else DocumentDomain.from_string(domain_value)
    )
    # Score may be ``_distance`` (L2) or ``_similarity`` depending on metric.
    score = row.get("_similarity")
    if score is None:
        # Convert L2 distance to a similarity proxy in [0, 1].
        dist = row.get("_distance", 0.0)
        score = max(0.0, 1.0 - dist)
    return RawHit(
        chunk_id=row.get("chunk_id", ""),
        content=row.get("content", ""),
        score=float(score),
        source_file=row.get("source_file", ""),
        date=parsed_date,
        domain=domain,
        tags=row.get("tags") or [],
        word_count=row.get("word_count", 0),
    )


__all__ = ["SemanticSearchEngine", "SearchOutcome"]
