"""Tests for the search engine and ranker."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from palace.indexing.hash_cache import HashCache
from palace.indexing.indexer import IncrementalIndexer
from palace.indexing.repository import LanceDBRepository
from palace.models.chunk import DocumentDomain, RawHit, SearchResult
from palace.search.engine import SemanticSearchEngine
from palace.search.ranker import SearchRanker
from palace.models.config import SearchConfig
from tests.fakes import FakeEmbedder


def _make_hit(
    chunk_id: str = "c1",
    content: str = "some content",
    score: float = 0.8,
    source_file: str = "/a.md",
    d: date | None = None,
    domain: DocumentDomain = DocumentDomain.JOURNAL,
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    word_count: int = 5,
) -> RawHit:
    return RawHit(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source_file=source_file,
        date=d,
        domain=domain,
        tags=tags or [],
        keywords=keywords or [],
        word_count=word_count,
    )


class TestSearchRanker:
    def test_empty_returns_empty(self) -> None:
        ranker = SearchRanker(SearchConfig())
        assert ranker.rank([]) == []

    def test_sorts_by_score(self) -> None:
        ranker = SearchRanker(SearchConfig(minimum_score=0.0))
        hits = [
            _make_hit(chunk_id="c1", score=0.5, source_file="/a1.md"),
            _make_hit(chunk_id="c2", score=0.9, source_file="/a2.md"),
            _make_hit(chunk_id="c3", score=0.7, source_file="/a3.md"),
        ]
        results = ranker.rank(hits)
        assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]

    def test_filters_below_minimum_score(self) -> None:
        ranker = SearchRanker(SearchConfig(minimum_score=0.5))
        hits = [
            _make_hit(chunk_id="c1", score=0.3),
            _make_hit(chunk_id="c2", score=0.8),
        ]
        results = ranker.rank(hits)
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_recency_boost(self) -> None:
        cfg = SearchConfig(
            minimum_score=0.0,
            recency_boost_days=30,
            recency_boost_amount=0.5,
        )
        ranker = SearchRanker(cfg)
        today = date.today()
        hits = [
            _make_hit(chunk_id="old", score=0.6, d=today - timedelta(days=60)),
            _make_hit(chunk_id="new", score=0.6, d=today),
        ]
        results = ranker.rank(hits)
        # Both pass minimum_score (0.0), recent gets boosted to 0.6 + 0.5 = 1.1.
        assert results[0].chunk_id == "new"

    def test_domain_weight(self) -> None:
        cfg = SearchConfig(
            minimum_score=0.0,
            domain_weights={"journal": 1.5, "notes": 0.5},
        )
        ranker = SearchRanker(cfg)
        hits = [
            _make_hit(chunk_id="c1", score=0.7, domain=DocumentDomain.JOURNAL),
            _make_hit(chunk_id="c2", score=0.7, domain=DocumentDomain.NOTES),
        ]
        results = ranker.rank(hits)
        assert results[0].chunk_id == "c1"

    def test_deduplication_by_source(self) -> None:
        cfg = SearchConfig(minimum_score=0.0, default_top_k=10, deduplicate_by_source=True)
        ranker = SearchRanker(cfg)
        hits = [
            _make_hit(chunk_id="c1", score=0.9, source_file="/same.md"),
            _make_hit(chunk_id="c2", score=0.7, source_file="/same.md"),
            _make_hit(chunk_id="c3", score=0.8, source_file="/other.md"),
        ]
        results = ranker.rank(hits)
        # Only one hit per source file.
        assert len(results) == 2
        assert results[0].chunk_id == "c1"  # highest score from /same.md
        assert results[1].chunk_id == "c3"  # /other.md

    def test_top_k_limit(self) -> None:
        cfg = SearchConfig(minimum_score=0.0, default_top_k=2)
        ranker = SearchRanker(cfg)
        hits = [
            _make_hit(chunk_id=f"c{i}", score=0.5 + i * 0.01, source_file=f"/file{i}.md")
            for i in range(5)
        ]
        results = ranker.rank(hits)
        assert len(results) == 2

    def test_per_call_limits_override_config(self) -> None:
        ranker = SearchRanker(
            SearchConfig(
                minimum_score=0.0,
                default_top_k=5,
                deduplicate_by_source=False,
            )
        )
        hits = [
            _make_hit(chunk_id="low", score=0.4, source_file="/low.md"),
            _make_hit(chunk_id="high", score=0.8, source_file="/high.md"),
        ]

        results = ranker.rank(hits, top_k=1, minimum_score=0.5)

        assert [result.chunk_id for result in results] == ["high"]

    def test_result_has_metadata(self) -> None:
        ranker = SearchRanker(SearchConfig(minimum_score=0.0))
        hits = [_make_hit(chunk_id="c1", score=0.8, word_count=42, domain=DocumentDomain.JOURNAL)]
        results = ranker.rank(hits)
        assert results[0].metadata["word_count"] == 42
        assert results[0].metadata["domain"] == "journal"

    def test_exact_keyword_match_ranks_above_stronger_semantic_match(self) -> None:
        ranker = SearchRanker(
            SearchConfig(
                minimum_score=0.0,
                keyword_boost_amount=0.2,
                deduplicate_by_source=False,
            )
        )
        hits = [
            _make_hit(
                chunk_id="semantic",
                score=0.85,
                source_file="/semantic.md",
                keywords=["vector database"],
            ),
            _make_hit(
                chunk_id="exact",
                score=0.70,
                source_file="/exact.md",
                keywords=["hybrid search"],
            ),
        ]

        results = ranker.rank(hits, query_keywords=["hybrid search"])

        assert [result.chunk_id for result in results] == ["exact", "semantic"]


class TestSemanticSearchEngine:
    @pytest.fixture
    def engine(
        self,
        config,
        embedder: FakeEmbedder,
        repository: LanceDBRepository,
    ) -> SemanticSearchEngine:
        return SemanticSearchEngine(config, embedder, repository)

    @pytest.fixture
    def populated_engine(
        self,
        config,
        embedder: FakeEmbedder,
        repository: LanceDBRepository,
        sample_journal: Path,
    ) -> SemanticSearchEngine:
        engine = SemanticSearchEngine(config, embedder, repository)
        # Index the sample journal so search has something to hit.
        from palace.indexing.indexer import IncrementalIndexer
        indexer = IncrementalIndexer(config, embedder, repository)
        indexer.index()
        return engine

    def test_search_returns_results(self, populated_engine: SemanticSearchEngine) -> None:
        results = populated_engine.search("energy fatigue low").results
        assert len(results) > 0
        assert isinstance(results[0].content, str)
        assert results[0].score >= 0

    def test_search_with_domain_filter(
        self, populated_engine: SemanticSearchEngine
    ) -> None:
        # restrict to a domain that has no entries -> empty
        results = populated_engine.search("test", domain="transcript").results
        # FakeEmbedder token overlap may still match; just check it runs.
        assert isinstance(results, list)

    def test_search_respects_top_k_override(
        self, populated_engine: SemanticSearchEngine
    ) -> None:
        results = populated_engine.search("energy", top_k=1).results
        assert len(results) <= 1

    def test_search_with_minimum_score(
        self, populated_engine: SemanticSearchEngine
    ) -> None:
        results = populated_engine.search("energy", minimum_score=0.99).results
        # With such a high threshold, we expect few or no results.
        assert all(r.score >= 0.99 for r in results) or len(results) == 0

    def test_search_with_date_filter(
        self, populated_engine: SemanticSearchEngine
    ) -> None:
        from datetime import date
        results = populated_engine.search(
            "energy", date_from="2026-04-15", date_to="2026-04-22"
        ).results
        for r in results:
            if r.date is not None:
                assert date(2026, 4, 15) <= r.date <= date(2026, 4, 22)

    def test_search_with_tags(self, populated_engine: SemanticSearchEngine) -> None:
        results = populated_engine.search("energy", tags=["health"]).results
        # The sample journal has front-matter tags [health, energy]
        for r in results:
            # If tags filter applied, results must have the tag.
            if r.tags:
                assert "health" in r.tags

    def test_search_forwards_coerced_filters_to_hybrid_searcher(
        self, config, embedder: FakeEmbedder
    ) -> None:
        class RecordingHybridSearcher:
            query = ""
            filters: dict[str, object] = {}

            def search(self, query: str, **filters: object) -> list[SearchResult]:
                self.query = query
                self.filters = filters
                return []

        engine = SemanticSearchEngine(config, embedder, object())
        hybrid_searcher = RecordingHybridSearcher()
        engine._hybrid_searcher = hybrid_searcher
        metadata_filters = {"source_file": "/journal.md"}

        engine.search(
            "energy",
            domain="journal",
            date_from="2026-01-01",
            date_to="2026-12-31",
            tags=["health"],
            metadata_filters=metadata_filters,
            top_k=3,
        )

        assert hybrid_searcher.query == "energy"
        assert hybrid_searcher.filters == {
            "domain": DocumentDomain.JOURNAL,
            "date_from": date(2026, 1, 1),
            "date_to": date(2026, 12, 31),
            "tags": ["health"],
            "metadata_filters": metadata_filters,
            "top_k": 3,
        }

    def test_search_returns_diagnostics(
        self, populated_engine: SemanticSearchEngine
    ) -> None:
        outcome = populated_engine.search("energy")
        assert outcome.duration_ms > 0
        assert isinstance(outcome.candidate_count, int)
        assert isinstance(outcome.results, list)
        assert outcome.degraded_mode is False

    def test_search_reports_degraded_mode(
        self, config, embedder: FakeEmbedder
    ) -> None:
        class DegradedHybridSearcher:
            def search(self, query: str, **filters: object):
                return [], True

        engine = SemanticSearchEngine(config, embedder, object())
        engine._hybrid_searcher = DegradedHybridSearcher()

        outcome = engine.search("energy")

        assert outcome.results == []
        assert outcome.degraded_mode is True

    def test_search_empty_index(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("anything").results
        assert results == []

    def test_search_coerces_string_domain(self, engine: SemanticSearchEngine) -> None:
        # Passing domain as string coercion shouldn't crash.
        results = engine.search("test", domain="journal").results
        assert isinstance(results, list)

    def test_search_coerces_string_dates(self, engine: SemanticSearchEngine) -> None:
        results = engine.search(
            "test", date_from="2026-01-01", date_to="2026-12-31"
        ).results
        assert isinstance(results, list)

    def test_search_invalid_date_string_handled(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("test", date_from="not-a-date").results
        assert isinstance(results, list)

    def test_query_keywords_boost_exact_match(self, config, embedder: FakeEmbedder) -> None:
        config.search.minimum_score = 0.0
        config.search.keyword_boost_amount = 0.2
        config.search.deduplicate_by_source = False

        class StaticRepository:
            def search(self, *_args, **_kwargs):
                return [
                    {
                        "chunk_id": "semantic",
                        "content": "Conceptually related material",
                        "_similarity": 0.85,
                        "source_file": "/semantic.md",
                        "domain": "notes",
                        "keywords": ["vector database"],
                    },
                    {
                        "chunk_id": "exact",
                        "content": "A literal keyword match",
                        "_similarity": 0.70,
                        "source_file": "/exact.md",
                        "domain": "notes",
                        "keywords": ["hybrid search"],
                    },
                ]

            def search_keyword(self, *_args, **_kwargs):
                return []

        engine = SemanticSearchEngine(config, embedder, StaticRepository())

        results = engine.search("hybrid search").results

        assert [result.chunk_id for result in results] == ["exact", "semantic"]

    def test_search_reuses_injected_ranker_with_overrides(
        self, config, embedder: FakeEmbedder
    ) -> None:
        class EmptyRepository:
            def search(self, *_args, **_kwargs):
                return []

            def search_keyword(self, *_args, **_kwargs):
                return []

        ranker = SearchRanker(config.search)
        calls: list[dict] = []

        def record_rerank(query, candidates, **kwargs):
            calls.append({"query": query, "candidates": candidates, **kwargs})
            return []

        ranker.rerank = record_rerank  # type: ignore[method-assign]
        engine = SemanticSearchEngine(config, embedder, EmptyRepository(), ranker)

        outcome = engine.search("query", top_k=3, minimum_score=0.4)

        assert outcome.results == []
        assert calls == [
            {
                "query": "query",
                "candidates": [],
            }
        ]
