"""Tests for the hash cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from palace.indexing.hash_cache import FileEntry, HashCache


class TestHashCache:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "cache.json")
        cache.load()
        assert len(cache) == 0
        assert cache.all_keys() == set()

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        p.write_text("{ not valid json }", encoding="utf-8")
        cache = HashCache(p)
        cache.load()  # should not raise
        assert len(cache) == 0

    def test_upsert_and_get(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "cache.json")
        cache.load()
        cache.upsert("file1", "hash1", 123.0, 5)
        entry = cache.get("file1")
        assert entry is not None
        assert entry.content_hash == "hash1"
        assert entry.chunk_count == 5

    def test_is_unchanged(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "cache.json")
        cache.load()
        cache.upsert("file1", "hash1", 123.0, 5)
        assert cache.is_unchanged("file1", "hash1") is True
        assert cache.is_unchanged("file1", "different") is False
        assert cache.is_unchanged("unknown", "hash1") is False

    def test_remove(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "cache.json")
        cache.load()
        cache.upsert("file1", "hash1", 123.0, 5)
        cache.remove("file1")
        assert not cache.has("file1")
        assert cache.get("file1") is None

    def test_save_and_reload(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = HashCache(p)
        cache.load()
        cache.upsert("file1", "hash1", 123.0, 5)
        cache.upsert("file2", "hash2", 456.0, 3)
        cache.save()
        assert p.exists()

        cache2 = HashCache(p)
        cache2.load()
        assert len(cache2) == 2
        assert cache2.get("file1").content_hash == "hash1"
        assert cache2.get("file2").chunk_count == 3

    def test_all_keys(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "cache.json")
        cache.load()
        cache.upsert("a", "h1", 1.0, 1)
        cache.upsert("b", "h2", 2.0, 2)
        keys = cache.all_keys()
        assert keys == {"a", "b"}

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        cache = HashCache(tmp_path / "nested" / "dir" / "cache.json")
        cache.load()
        cache.upsert("file1", "hash1", 1.0, 1)
        cache.save()
        assert (tmp_path / "nested" / "dir" / "cache.json").exists()
