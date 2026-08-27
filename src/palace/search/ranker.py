"""Unified ranking for vector-only and hybrid retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from palace.models.chunk import RawHit, SearchResult
from palace.models.config import SearchConfig


@dataclass
class RerankConfig:
    """Configuration for hybrid fusion and cross-encoder scoring."""

    retrieval_top_k: int = 20
    final_top_n: int = 5
    rrf_k: float = 60.0
    cross_encoder_weight: float = 0.7
    minimum_score: float = 0.0


class CrossEncoderModel:
    """Cross-encoder interface with a lightweight local default scorer."""

    def score(self, query: str, content: str, base_score: float) -> float:
        """Score a query and candidate together."""
        query_terms = {
            term.strip().casefold()
            for term in query.split()
            if len(term.strip()) > 2
        }
        if not query_terms:
            return base_score

        content_lower = content.casefold()
        overlap = sum(term in content_lower for term in query_terms)
        overlap_ratio = overlap / len(query_terms)
        cross_encoder_score = min(1.0, overlap_ratio * 1.5)
        return 0.3 * base_score + 0.7 * cross_encoder_score


@dataclass
class HybridRetrievalResult:
    """One ranked candidate from a vector or keyword retriever."""

    hit: RawHit
    source: str
    raw_rank: int
    raw_score: float


class RRFRescorer:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: float = 60.0) -> None:
        self._k = k

    def fuse(
        self,
        results_by_source: list[list[HybridRetrievalResult]],
    ) -> list[HybridRetrievalResult]:
        """Return one fused candidate per stable chunk ID."""
        fused_scores: dict[str, float] = {}
        representatives: dict[str, HybridRetrievalResult] = {}
        sources: dict[str, list[str]] = {}

        for ranked_list in results_by_source:
            for position, result in enumerate(ranked_list, start=1):
                key = _candidate_key(result)
                fused_scores[key] = (
                    fused_scores.get(key, 0.0) + 1.0 / (self._k + position)
                )
                representatives.setdefault(key, result)
                candidate_sources = sources.setdefault(key, [])
                if result.source not in candidate_sources:
                    candidate_sources.append(result.source)

        fused = [
            HybridRetrievalResult(
                hit=representatives[key].hit,
                source="+".join(sources[key]),
                raw_rank=representatives[key].raw_rank,
                raw_score=score,
            )
            for key, score in fused_scores.items()
        ]
        fused.sort(key=lambda result: result.raw_score, reverse=True)
        return fused


class SearchRanker:
    """Apply retrieval fusion, relevance, policy signals, and deduplication."""

    def __init__(
        self,
        config: SearchConfig,
        rerank_config: RerankConfig | None = None,
        cross_encoder: CrossEncoderModel | None = None,
    ) -> None:
        self._config = config
        self._rerank_config = rerank_config or RerankConfig()
        self._cross_encoder = cross_encoder or CrossEncoderModel()
        self._rrf = RRFRescorer(self._rerank_config.rrf_k)

    def rank(
        self,
        hits: list[RawHit],
        as_of: datetime | None = None,
        query_keywords: list[str] | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        """Rank a single list of retrieval hits."""
        if not hits:
            return []

        now = as_of or datetime.utcnow()
        result_limit = top_k if top_k is not None else self._config.default_top_k
        threshold = (
            minimum_score
            if minimum_score is not None
            else self._config.minimum_score
        )
        normalized_keywords = _normalize_keywords(query_keywords)
        scored = []
        for hit in hits:
            score = self._apply_policy_score(
                hit,
                hit.score,
                now=now,
                query_keywords=normalized_keywords,
            )
            if score >= threshold:
                scored.append((score, hit, None))

        return self._finalize(scored, result_limit)

    def rerank(
        self,
        query: str,
        hybrid_results: list[HybridRetrievalResult],
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        """Fuse and rerank vector and keyword candidates."""
        if not hybrid_results:
            return []

        results_by_source: dict[str, list[HybridRetrievalResult]] = {}
        for result in hybrid_results:
            results_by_source.setdefault(result.source, []).append(result)
        for results in results_by_source.values():
            results.sort(key=lambda result: result.raw_rank)

        fused = self._rrf.fuse(list(results_by_source.values()))
        candidates = fused[: self._rerank_config.retrieval_top_k]
        threshold = (
            minimum_score
            if minimum_score is not None
            else self._rerank_config.minimum_score
        )
        now = datetime.utcnow()
        query_keywords = _normalize_keywords([query])
        scored = []
        for candidate in candidates:
            cross_encoder_score = self._cross_encoder.score(
                query,
                candidate.hit.content,
                candidate.hit.score,
            )
            score = (
                self._rerank_config.cross_encoder_weight * cross_encoder_score
                + (1 - self._rerank_config.cross_encoder_weight)
                * candidate.raw_score
            )
            score = self._apply_policy_score(
                candidate.hit,
                score,
                now=now,
                query_keywords=query_keywords,
            )
            if score >= threshold:
                scored.append((score, candidate.hit, candidate.source))

        return self._finalize(scored, self._rerank_config.final_top_n)

    def _apply_policy_score(
        self,
        hit: RawHit,
        score: float,
        *,
        now: datetime,
        query_keywords: set[str] | None = None,
    ) -> float:
        """Apply keyword, recency, and domain policy signals to a score."""
        if query_keywords:
            hit_keywords = _normalize_keywords(hit.keywords)
            matching_keywords = query_keywords & hit_keywords
            score += self._config.keyword_boost_amount * len(matching_keywords)

        if hit.date is not None:
            age_days = (now.date() - hit.date).days
            if age_days <= self._config.recency_boost_days:
                score += self._config.recency_boost_amount * (
                    1 - age_days / max(1, self._config.recency_boost_days)
                )

        return score * self._config.domain_weights.get(hit.domain.value, 1.0)

    def _finalize(
        self,
        scored: list[tuple[float, RawHit, str | None]],
        result_limit: int,
    ) -> list[SearchResult]:
        """Sort, deduplicate, limit, and serialize scored hits."""
        scored.sort(key=lambda item: item[0], reverse=True)
        if self._config.deduplicate_by_source:
            scored = self._deduplicate(scored)

        return [
            _to_search_result(score, hit, retrieval_source)
            for score, hit, retrieval_source in scored[:result_limit]
        ]

    @staticmethod
    def _deduplicate(
        scored: list[tuple[float, RawHit, str | None]],
    ) -> list[tuple[float, RawHit, str | None]]:
        """Keep the highest-scoring result from each source file."""
        seen_sources: set[str] = set()
        deduplicated = []
        for item in scored:
            source_file = item[1].source_file
            if source_file not in seen_sources:
                seen_sources.add(source_file)
                deduplicated.append(item)
        return deduplicated


def _candidate_key(result: HybridRetrievalResult) -> str:
    """Return the stable identity used to fuse candidates across retrievers."""
    if result.hit.chunk_id:
        return result.hit.chunk_id
    return f"{result.hit.source_file}\0{result.hit.content}"


def _normalize_keywords(keywords: list[str] | None) -> set[str]:
    return {
        keyword.casefold().strip()
        for keyword in keywords or []
        if keyword.strip()
    }


def _to_search_result(
    score: float,
    hit: RawHit,
    retrieval_source: str | None,
) -> SearchResult:
    metadata = {
        "word_count": hit.word_count,
        "domain": hit.domain.value,
        "keywords": hit.keywords,
    }
    if retrieval_source is not None:
        metadata["retrieval_source"] = retrieval_source

    return SearchResult(
        content=hit.content,
        score=round(score, 4),
        source_file=hit.source_file,
        date=hit.date,
        domain=hit.domain,
        tags=hit.tags,
        chunk_id=hit.chunk_id,
        metadata=metadata,
    )


__all__ = [
    "CrossEncoderModel",
    "HybridRetrievalResult",
    "RerankConfig",
    "RRFRescorer",
    "SearchRanker",
]
