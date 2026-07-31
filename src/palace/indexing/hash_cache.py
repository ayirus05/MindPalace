"""JSON-backed hash cache for incremental indexing.

Persists a mapping ``source_file -> {hash, mtime, chunk_count, last_indexed_at}``
so the indexer can skip unchanged files on subsequent runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


logger = logging.getLogger("palace.indexing.cache")


class FileEntry(BaseModel):
    """One cached file record."""

    content_hash: str
    mtime: float
    chunk_count: int
    last_indexed_at: datetime


class HashCache:
    """A small JSON-backed cache of file hashes.

    Load/save roundtrips through a single JSON file.  In-memory state is a
    plain dict so reads are cheap during indexing.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, FileEntry] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """Load the cache from disk.  Missing or corrupt file is treated as empty."""
        if self._loaded:
            return
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for key, val in raw.items():
                    self._entries[key] = FileEntry.model_validate(val)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Hash cache at %s is corrupt (%s); starting fresh", self._path, exc
                )
                self._entries = {}
        self._loaded = True

    def save(self) -> None:
        """Persist the cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved hash cache with %d entries to %s", len(self._entries), self._path)

    def has(self, source_file: str) -> bool:
        return source_file in self._entries

    def get(self, source_file: str) -> FileEntry | None:
        return self._entries.get(source_file)

    def is_unchanged(self, source_file: str, content_hash: str) -> bool:
        entry = self._entries.get(source_file)
        return entry is not None and entry.content_hash == content_hash

    def upsert(
        self,
        source_file: str,
        content_hash: str,
        mtime: float,
        chunk_count: int,
    ) -> None:
        self._entries[source_file] = FileEntry(
            content_hash=content_hash,
            mtime=mtime,
            chunk_count=chunk_count,
            last_indexed_at=datetime.utcnow(),
        )

    def remove(self, source_file: str) -> None:
        self._entries.pop(source_file, None)

    def all_keys(self) -> set[str]:
        return set(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["HashCache", "FileEntry"]
