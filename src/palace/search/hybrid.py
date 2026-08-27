"""Hybrid dense-vector and BM25 keyword search orchestration."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from palace.indexing.repository import LanceDBRepository
from palace.models.chunk import DocumentDomain, RawHit, SearchResult
from palace.search.ranker import HybridRetrievalResult, SearchRanker


class EmbeddingModel(Protocol):
    """Minimal query-embedding contract required by hybrid search."""

    def embed_one(self, text: str) -> list[float]:
        """Generate one dense embedding vector."""
        ...


class HybridSearcher:
    """Retrieve dense and lexical candidates, then rerank them together."""

    def __init__(
        self,
        repository: LanceDBRepository,
        embedding_model: EmbeddingModel,
        ranker: SearchRanker,
    ) -> None:
        self._repository = repository
        self._embedding_model = embedding_model
        self._ranker = ranker

    def search(self, query: str, **filters: Any) -> list[SearchResult]:
        """Run vector and FTS retrieval and return the reranked results."""
        query_vector = self._embedding_model.embed_one(query)

        vector_rows = self._repository.search(query_vector, **filters)
        vector_results = _to_hybrid_results(vector_rows, source="vector")

        keyword_rows = self._repository.search_keyword(query, **filters)
        keyword_results = _to_hybrid_results(keyword_rows, source="fts")

        return self._ranker.rerank(
            query,
            [*vector_results, *keyword_results],
        )


def _to_hybrid_results(
    rows: list[dict[str, Any]], source: str
) -> list[HybridRetrievalResult]:
    """Convert repository rows into ranked hybrid retrieval candidates."""
    results: list[HybridRetrievalResult] = []
    for raw_rank, row in enumerate(rows, start=1):
        hit = _row_to_hit(row)
        results.append(
            HybridRetrievalResult(
                hit=hit,
                source=source,
                raw_rank=raw_rank,
                raw_score=hit.score,
            )
        )
    return results


def _row_to_hit(row: dict[str, Any]) -> RawHit:
    """Normalize a vector or FTS repository row into a ``RawHit``."""
    row_date = row.get("date")
    if isinstance(row_date, date):
        parsed_date = row_date
    elif row_date:
        try:
            parsed_date = date.fromisoformat(str(row_date))
        except ValueError:
            parsed_date = None
    else:
        parsed_date = None

    domain_value = row.get("domain")
    domain = (
        domain_value
        if isinstance(domain_value, DocumentDomain)
        else DocumentDomain.from_string(domain_value)
    )

    if row.get("_similarity") is not None:
        score = float(row["_similarity"])
    elif row.get("_score") is not None:
        score = float(row["_score"])
    elif row.get("_distance") is not None:
        score = 1.0 / (1.0 + float(row["_distance"]))
    else:
        score = 0.0

    return RawHit(
        chunk_id=row.get("chunk_id", ""),
        content=row.get("content", ""),
        score=score,
        source_file=row.get("source_file", ""),
        date=parsed_date,
        domain=domain,
        tags=row.get("tags") or [],
        keywords=row.get("keywords") or [],
        word_count=row.get("word_count", 0),
    )


__all__ = ["EmbeddingModel", "HybridSearcher"]
