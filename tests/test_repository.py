"""Tests for the LanceDB repository layer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from palace.indexing.repository import LanceDBRepository
from palace.models.chunk import Chunk, ChunkMetadata, DocumentDomain


def _make_chunk(
    chunk_id: str,
    source_file: str,
    content: str,
    embedding: list[float],
    domain: DocumentDomain = DocumentDomain.JOURNAL,
    d: date | None = None,
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
) -> Chunk:
    word_count = len(content.split())
    md = ChunkMetadata(
        chunk_id=chunk_id,
        source_file=source_file,
        source_file_hash="src_hash",
        domain=domain,
        date=d,
        tags=tags or [],
        keywords=keywords or [],
        content_hash="content_hash",
        word_count=word_count,
        token_estimate=int(round(word_count * 1.3)),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    return Chunk(metadata=md, content=content, embedding=embedding)


class TestLanceDBRepository:
    def test_insert_and_count(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "hello world", [0.1, 0.2]),
            _make_chunk("c2", "/a.md", "second chunk", [0.3, 0.4]),
        ]
        n = repository.insert_chunks(chunks)
        assert n == 2
        assert repository.count() == 2

    def test_insert_persists_keywords(self, repository: LanceDBRepository) -> None:
        chunk = _make_chunk(
            "c1", "/a.md", "hybrid search", [0.1, 0.2], keywords=["hybrid search"]
        )
        repository.insert_chunks([chunk])

        result = repository.search(query_vector=[0.1, 0.2], top_k=1)

        assert result[0]["keywords"] == ["hybrid search"]

    def test_insert_empty(self, repository: LanceDBRepository) -> None:
        assert repository.insert_chunks([]) == 0

    def test_insert_without_embedding_raises(self, repository: LanceDBRepository) -> None:
        chunk = _make_chunk("c1", "/a.md", "hello", [])
        with pytest.raises(ValueError):
            repository.insert_chunks([chunk])

    def test_delete_chunks(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "hello world", [0.1, 0.2]),
            _make_chunk("c2", "/b.md", "other file", [0.3, 0.4]),
        ]
        repository.insert_chunks(chunks)
        deleted = repository.delete_chunks("/a.md")
        assert deleted == 1
        assert repository.count() == 1

    def test_update_chunks_replaces(self, repository: LanceDBRepository) -> None:
        original = [
            _make_chunk("c1", "/a.md", "old content", [0.1, 0.2]),
            _make_chunk("c2", "/a.md", "old content 2", [0.3, 0.4]),
        ]
        repository.insert_chunks(original)
        assert repository.count() == 2

        updated = [
            _make_chunk("c3", "/a.md", "new content", [0.5, 0.6]),
        ]
        n = repository.update_chunks("/a.md", updated)
        assert n == 1
        assert repository.count() == 1

    def test_reopened_table_supports_insert_and_update(self, config) -> None:
        db_path = config.resolve(config.database.path)
        first = LanceDBRepository(db_path, config.database.table_name)
        first.insert_chunks([
            _make_chunk("c1", "/a.md", "first", [0.1, 0.2])
        ])

        reopened = LanceDBRepository(db_path, config.database.table_name)
        reopened.insert_chunks([
            _make_chunk("c2", "/b.md", "second", [0.3, 0.4])
        ])
        reopened.update_chunks(
            "/a.md",
            [_make_chunk("c3", "/a.md", "replacement", [0.5, 0.6])],
        )

        assert reopened.count() == 2
        assert reopened.filter_by_metadata(source_file="/a.md")[0]["chunk_id"] == "c3"

    def test_search_returns_results(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "hello world", [0.1, 0.2]),
            _make_chunk("c2", "/b.md", "other file", [0.9, 0.8]),
        ]
        repository.insert_chunks(chunks)
        results = repository.search(query_vector=[0.9, 0.8], top_k=2)
        assert len(results) == 2
        # Closest vector should come first.
        top = results[0]
        assert top["source_file"] == "/b.md"

    def test_search_with_domain_filter(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "journal entry", [0.1, 0.2], domain=DocumentDomain.JOURNAL),
            _make_chunk("c2", "/b.md", "random note", [0.9, 0.8], domain=DocumentDomain.NOTES),
        ]
        repository.insert_chunks(chunks)
        results = repository.search(
            query_vector=[0.1, 0.2],
            top_k=5,
            domain=DocumentDomain.JOURNAL,
        )
        assert len(results) == 1
        assert results[0]["source_file"] == "/a.md"

    def test_search_with_date_filter(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "old entry", [0.1, 0.2], d=date(2026, 1, 1)),
            _make_chunk("c2", "/b.md", "new entry", [0.1, 0.2], d=date(2026, 6, 1)),
        ]
        repository.insert_chunks(chunks)
        results = repository.search(
            query_vector=[0.1, 0.2],
            top_k=5,
            date_from=date(2026, 5, 1),
        )
        assert len(results) == 1
        assert results[0]["date"] == "2026-06-01"

    def test_search_with_tags(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "tagged entry", [0.1, 0.2], tags=["health", "energy"]),
            _make_chunk("c2", "/b.md", "untagged entry", [0.9, 0.8], tags=[]),
        ]
        repository.insert_chunks(chunks)
        results = repository.search(
            query_vector=[0.1, 0.2],
            top_k=5,
            tags=["health"],
        )
        assert len(results) == 1
        assert results[0]["source_file"] == "/a.md"

    def test_search_with_metadata_filters(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "first", [0.1, 0.2]),
            _make_chunk("c2", "/b.md", "second", [0.1, 0.2]),
        ]
        repository.insert_chunks(chunks)

        results = repository.search(
            query_vector=[0.1, 0.2],
            top_k=5,
            metadata_filters={"source_file": "/b.md"},
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    def test_search_rejects_invalid_metadata_filter_key(
        self, repository: LanceDBRepository
    ) -> None:
        repository.insert_chunks([
            _make_chunk("c1", "/a.md", "first", [0.1, 0.2]),
        ])

        with pytest.raises(ValueError, match="Invalid metadata filter key"):
            repository.search(
                query_vector=[0.1, 0.2],
                metadata_filters={"source_file = '/a.md' OR 1": "/b.md"},
            )

    def test_filter_by_metadata(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "content one", [0.1, 0.2]),
            _make_chunk("c2", "/b.md", "content two", [0.3, 0.4]),
        ]
        repository.insert_chunks(chunks)
        rows = repository.filter_by_metadata(source_file="/a.md")
        assert len(rows) == 1
        assert rows[0]["chunk_id"] == "c1"

    def test_statistics(self, repository: LanceDBRepository) -> None:
        chunks = [
            _make_chunk("c1", "/a.md", "content", [0.1, 0.2], domain=DocumentDomain.JOURNAL),
            _make_chunk("c2", "/b.md", "more content", [0.3, 0.4], domain=DocumentDomain.NOTES),
            _make_chunk("c3", "/b.md", "third", [0.5, 0.6], domain=DocumentDomain.NOTES, d=date(2026, 4, 1)),
        ]
        repository.insert_chunks(chunks)
        stats = repository.statistics()
        assert stats["total_chunks"] == 3
        assert stats["unique_source_files"] == 2
        assert stats["by_domain"]["journal"] == 1
        assert stats["by_domain"]["notes"] == 2
        assert stats["date_range_earliest"] == "2026-04-01"

    def test_vacuum_returns_dict(self, repository: LanceDBRepository) -> None:
        chunks = [_make_chunk("c1", "/a.md", "content", [0.1, 0.2])]
        repository.insert_chunks(chunks)
        result = repository.vacuum()
        assert "action" in result
        assert "rows_before" in result
        assert "rows_after" in result

    def test_count_empty(self, repository: LanceDBRepository) -> None:
        assert repository.count() == 0
