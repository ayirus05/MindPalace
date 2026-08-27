"""Tests for hybrid dense and keyword search orchestration."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from palace.models.chunk import DocumentDomain, RawHit, SearchResult
from palace.models.config import SearchConfig
from palace.search.hybrid import HybridSearcher
from palace.search.ranker import (
    CrossEncoderModel,
    HybridRetrievalResult,
    RerankConfig,
    SearchRanker,
)


class RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class RecordingRepository:
    def __init__(self) -> None:
        self.vector_calls: list[tuple[list[float], dict[str, Any]]] = []
        self.keyword_calls: list[tuple[str, dict[str, Any]]] = []

    def search(
        self, vector: list[float], **filters: Any
    ) -> list[dict[str, Any]]:
        self.vector_calls.append((vector, filters))
        return [
            {
                "chunk_id": "vector-1",
                "content": "dense result",
                "source_file": "/vector.md",
                "domain": DocumentDomain.NOTES,
                "date": "2026-08-20",
                "_distance": 0.25,
            },
            {
                "chunk_id": "vector-2",
                "content": "second dense result",
                "source_file": "/vector-2.md",
                "domain": "journal",
                "_similarity": 0.7,
            },
        ]

    def search_keyword(
        self, query: str, **filters: Any
    ) -> list[dict[str, Any]]:
        self.keyword_calls.append((query, filters))
        return [
            {
                "chunk_id": "fts-1",
                "content": "keyword result",
                "source_file": "/keyword.md",
                "domain": "notes",
                "_score": 2.5,
            }
        ]


class RecordingRanker:
    def __init__(self) -> None:
        self.query = ""
        self.candidates: list[HybridRetrievalResult] = []
        self.results = [
            SearchResult(
                content="final result",
                score=0.9,
                source_file="/final.md",
                date=None,
                domain=DocumentDomain.NOTES,
                tags=[],
                chunk_id="final",
            )
        ]

    def rerank(
        self, query: str, candidates: list[HybridRetrievalResult]
    ) -> list[SearchResult]:
        self.query = query
        self.candidates = candidates
        return self.results


def test_search_combines_vector_and_fts_candidates_before_reranking() -> None:
    repository = RecordingRepository()
    embedder = RecordingEmbedder()
    ranker = RecordingRanker()
    searcher = HybridSearcher(repository, embedder, ranker)
    filters = {"top_k": 10, "tags": ["search"]}

    results = searcher.search("hybrid retrieval", **filters)

    assert results == ranker.results
    assert embedder.queries == ["hybrid retrieval"]
    assert repository.vector_calls == [([0.1, 0.2], filters)]
    assert repository.keyword_calls == [("hybrid retrieval", filters)]
    assert ranker.query == "hybrid retrieval"
    assert [candidate.source for candidate in ranker.candidates] == [
        "vector",
        "vector",
        "fts",
    ]
    assert [candidate.raw_rank for candidate in ranker.candidates] == [1, 2, 1]
    assert [candidate.raw_score for candidate in ranker.candidates] == [0.8, 0.7, 2.5]
    assert ranker.candidates[0].hit.date.isoformat() == "2026-08-20"
    assert ranker.candidates[1].hit.domain == DocumentDomain.JOURNAL


class RecordingCrossEncoder(CrossEncoderModel):
    def __init__(self) -> None:
        self.chunk_contents: list[str] = []

    def score(self, query: str, content: str, base_score: float) -> float:
        self.chunk_contents.append(content)
        return base_score


def _hit(
    chunk_id: str,
    source_file: str,
    score: float,
    *,
    domain: DocumentDomain = DocumentDomain.JOURNAL,
    result_date: date | None = None,
) -> RawHit:
    return RawHit(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        score=score,
        source_file=source_file,
        date=result_date,
        domain=domain,
    )


def test_unified_ranker_combines_hybrid_and_policy_signals() -> None:
    cross_encoder = RecordingCrossEncoder()
    ranker = SearchRanker(
        SearchConfig(
            minimum_score=0.0,
            recency_boost_days=30,
            recency_boost_amount=0.5,
            domain_weights={"notes": 2.0},
            deduplicate_by_source=True,
        ),
        RerankConfig(final_top_n=10),
        cross_encoder,
    )
    recent = _hit(
        "shared",
        "/same.md",
        0.4,
        domain=DocumentDomain.NOTES,
        result_date=date.today(),
    )
    candidates = [
        HybridRetrievalResult(recent, "vector", 1, 0.4),
        HybridRetrievalResult(recent.model_copy(), "fts", 1, 2.0),
        HybridRetrievalResult(
            _hit("old", "/same.md", 0.9, result_date=date.today() - timedelta(days=90)),
            "vector",
            2,
            0.9,
        ),
        HybridRetrievalResult(_hit("other", "/other.md", 0.2), "fts", 2, 0.2),
    ]

    results = ranker.rerank("content", candidates)

    assert [result.chunk_id for result in results] == ["shared", "other"]
    assert results[0].metadata["retrieval_source"] == "vector+fts"
    assert sorted(cross_encoder.chunk_contents) == [
        "content for old",
        "content for other",
        "content for shared",
    ]
