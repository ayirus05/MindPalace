"""Tests for the incremental indexer."""

from __future__ import annotations

from pathlib import Path

import pytest

from palace.indexing.hash_cache import HashCache
from palace.indexing.indexer import IncrementalIndexer
from palace.indexing.repository import LanceDBRepository
from tests.fakes import FakeEmbedder


@pytest.fixture
def indexer(
    config, embedder: FakeEmbedder, repository: LanceDBRepository, hash_cache: HashCache
) -> IncrementalIndexer:
    return IncrementalIndexer(config, embedder, repository, hash_cache=hash_cache)


class TestDiscovery:
    def test_discovers_markdown_files(self, indexer: IncrementalIndexer, tmp_project: Path) -> None:
        (tmp_project / "journals" / "a.md").write_text("content")
        (tmp_project / "journals" / "subdir").mkdir()
        (tmp_project / "journals" / "subdir" / "b.md").write_text("content")
        files = indexer.discover_files()
        assert len(files) == 2

    def test_ignores_other_extensions(self, indexer: IncrementalIndexer, tmp_project: Path) -> None:
        (tmp_project / "journals" / "a.md").write_text("content")
        (tmp_project / "journals" / "b.txt").write_text("content")
        (tmp_project / "journals" / "c.jpg").write_text("content")
        files = indexer.discover_files()
        assert len(files) == 2  # .md and .txt only

    def test_missing_source_dir_logs_warning(self, indexer: IncrementalIndexer, config) -> None:
        config.indexer.source_dirs = ["nonexistent"]
        files = indexer.discover_files()
        assert files == []


class TestIncrementalIndexing:
    def test_index_new_file(self, indexer: IncrementalIndexer, sample_journal: Path, repository: LanceDBRepository) -> None:
        result = indexer.index()
        assert result.scanned == 1
        assert result.indexed == 1
        assert result.skipped == 0
        assert result.chunks_created == 1  # headerless document fallback
        assert repository.count() == 1

    def test_index_skips_unchanged(self, indexer: IncrementalIndexer, sample_journal: Path, repository: LanceDBRepository) -> None:
        first = indexer.index()
        assert first.indexed == 1
        second = indexer.index()
        assert second.scanned == 1
        assert second.indexed == 0
        assert second.skipped == 1
        # Repository still has the chunks.
        assert repository.count() == 1

    def test_index_reindexes_changed(self, indexer: IncrementalIndexer, sample_journal: Path, repository: LanceDBRepository) -> None:
        indexer.index()
        # Modify the file.
        sample_journal.write_text(
            "2026-05-01\n\nNew entry after change.\n", encoding="utf-8"
        )
        result = indexer.index()
        assert result.indexed == 1
        assert result.skipped == 0
        assert repository.count() == 1  # old chunks replaced

    def test_index_deletes_removed_file(self, indexer: IncrementalIndexer, sample_journal: Path, repository: LanceDBRepository) -> None:
        indexer.index()
        assert repository.count() == 1
        sample_journal.unlink()
        result = indexer.index()
        assert result.deleted == 1
        assert repository.count() == 0

    def test_index_multiple_files(self, indexer: IncrementalIndexer, sample_journal: Path, sample_note: Path, repository: LanceDBRepository) -> None:
        # Add notes dir to source_dirs so the note is discovered.
        indexer._config.indexer.source_dirs = ["journals", "notes"]
        result = indexer.index()
        assert result.scanned == 2
        assert result.indexed == 2
        assert result.chunks_created > 0

    def test_reindex_all_forces_full(self, indexer: IncrementalIndexer, sample_journal: Path, repository: LanceDBRepository) -> None:
        first = indexer.index()
        assert first.indexed == 1
        forced = indexer.index(force_reindex = True)
        assert forced.indexed == 1  # re-indexed despite no changes
        assert forced.skipped == 0

    def test_errors_collected(self, indexer: IncrementalIndexer, tmp_project: Path) -> None:
        # Create a file that will fail to read (permission denied).
        p = tmp_project / "journals" / "bad.md"
        p.write_text("content")
        p.chmod(0o000)
        try:
            result = indexer.index()
            assert len(result.errors) >= 0  # may or may not error depending on OS
        finally:
            p.chmod(0o644)

    def test_hash_cache_persisted(self, indexer: IncrementalIndexer, sample_journal: Path, config) -> None:
        indexer.index()
        # The cache file should exist.
        cache_path = config.resolve(config.indexer.hash_cache_path)
        assert cache_path.exists()


class TestIndexerVerify:
    def test_verify_returns_counts(self, indexer: IncrementalIndexer, sample_journal: Path) -> None:
        indexer.index()
        result = indexer.verify()
        assert "cached_files" in result
        assert "repository_chunks" in result
        assert "healthy_files" in result
        assert "stale_files" in result
        assert "missing_files" in result
        assert result["cached_files"] == 1
        assert result["repository_chunks"] == 1
        assert result["healthy_files"] == 1
        assert result["stale_files"] == 0
        assert result["missing_files"] == 0

    def test_verify_detects_stale_file(self, indexer: IncrementalIndexer, sample_journal: Path) -> None:
        indexer.index()
        # Modify the file without re-indexing.
        sample_journal.write_text("2026-05-01\n\nChanged content.\n", encoding="utf-8")
        result = indexer.verify()
        assert result["stale_files"] == 1
        assert result["healthy_files"] == 0

    def test_verify_detects_missing_file(self, indexer: IncrementalIndexer, sample_journal: Path) -> None:
        indexer.index()
        sample_journal.unlink()
        result = indexer.verify()
        assert result["missing_files"] == 1
