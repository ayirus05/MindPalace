"""Search ranking: combine similarity, recency, domain, and dedup signals.

The :class:`SearchRanker` is decoupled from the repository.  It takes raw
hits (already scored by the vector store) and returns a re-ranked list that
surfaces the most *useful* results rather than only the highest cosine.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from palace.models.chunk import RawHit, SearchResult
from palace.models.config import SearchConfig


logger = logging.getLogger("palace.search.ranker")


class SearchRanker:
    """Re-rank raw vector hits into final :class:`SearchResult` list.

    Signals combined:
      * Semantic similarity (the base score from the vector store).
      * Recency: a small bonus for results within ``recency_boost_days``.
      * Domain preference: multiply by a per-domain weight (default 1.0).
      * Duplicate suppression: if two hits come from the same source file,
        keep only the best one (configurable).
      * Minimum-score threshold: drop anything below ``minimum_score``.
    """

    def __init__(self, config: SearchConfig) -> None:
        self._config = config

    def rank(
        self,
        hits: list[RawHit],
        as_of: datetime | None = None,
    ) -> list[SearchResult]:
        if not hits:
            return []
        now = as_of or datetime.utcnow()
        scored: list[tuple[float, RawHit]] = []
        for hit in hits:
            score = hit.score
            # Recency boost.
            if hit.date is not None:
                age_days = (now.date() - hit.date).days
                if age_days <= self._config.recency_boost_days:
                    boost = self._config.recency_boost_amount * (
                        1 - age_days / max(1, self._config.recency_boost_days)
                    )
                    score += boost
            # Domain weight.
            domain_weight = self._config.domain_weights.get(hit.domain.value, 1.0)
            score *= domain_weight
            if score >= self._config.minimum_score:
                scored.append((score, hit))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        if self._config.deduplicate_by_source:
            scored = self._deduplicate(scored)

        top = scored[: self._config.default_top_k]
        return [
            SearchResult(
                content=h.content,
                score=round(s, 4),
                source_file=h.source_file,
                date=h.date,
                domain=h.domain,
                tags=h.tags,
                chunk_id=h.chunk_id,
                metadata={
                    "word_count": h.word_count,
                    "domain": h.domain.value,
                },
            )
            for s, h in top
        ]

    def _deduplicate(
        self, scored: list[tuple[float, RawHit]]
    ) -> list[tuple[float, RawHit]]:
        """Keep only the best-scoring hit per source_file."""
        best_per_source: dict[str, tuple[float, RawHit]] = {}
        order: list[str] = []
        for score, hit in scored:
            if hit.source_file not in best_per_source:
                best_per_source[hit.source_file] = (score, hit)
                order.append(hit.source_file)
            # Keep insertion order; first-seen wins because list is already sorted desc.
        return [best_per_source[src] for src in order]
