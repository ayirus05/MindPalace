"""Incremental indexing pipeline.

The :class:`IncrementalIndexer` walks configured source directories, hashes
every file, and only re-embeds files whose content hash changed since the
last run.  Removed files are purged from the repository.  Unchanged files
are skipped entirely — no embedding work, no DB writes.

This is the hot path for production use: the goal is that re-running
``palace index`` after touching a handful of journal entries is cheap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from palace.embeddings.manager import Embedder
from palace.indexing.chunker import MarkdownASTChunker
from palace.indexing.hash_cache import HashCache
from palace.indexing.repository import ChunkRepository
from palace.metadata.extractor import MetadataExtractor
from palace.models.chunk import Chunk
from palace.models.config import PalaceConfig


logger = logging.getLogger("palace.indexer")


@dataclass
class IndexResult:
    """Summary of a single indexing run — surfaced to the CLI and logs."""

    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IncrementalIndexer:
    """Orchestrates the chunk → embed → upsert pipeline with delta detection."""

    def __init__(
        self,
        config: PalaceConfig,
        embedder: Embedder,
        repository: ChunkRepository,
        hash_cache: HashCache | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._repository = repository
        self._extractor = MetadataExtractor(config.chunker)
        self._hash_cache = hash_cache or HashCache(
            config.resolve(config.indexer.hash_cache_path)
        )
        # Extension allowlist, normalized to lowercase.
        self._extensions = {e.lower() for e in config.indexer.extensions}

    # -- discovery --

    def discover_files(self) -> list[Path]:
        """Walk configured source directories and return indexable files."""
        roots = self._config.resolve_all(self._config.indexer.source_dirs)
        found: list[Path] = []
        for root in roots:
            if not root.exists():
                logger.warning("Source directory does not exist: %s", root)
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in self._extensions:
                    found.append(path)
        return found

    # -- main pipeline --

    def index(self) -> IndexResult:
        """Run a full incremental index pass."""
        self._hash_cache.load()
        result = IndexResult(started_at=datetime.utcnow())
        t0 = time.monotonic()

        files = self.discover_files()
        result.scanned = len(files)
        logger.info("Discovered %d files to consider", len(files))

        seen_files: set[str] = set()
        for path in files:
            try:
                self._index_one(path, result, seen_files)
            except Exception as exc:
                msg = f"Failed to index {path}: {exc}"
                logger.exception(msg)
                result.errors.append(msg)

        # Delete files that have vanished from the source tree.
        for cached_key in self._hash_cache.all_keys():
            if cached_key not in seen_files:
                deleted = self._repository.delete_chunks(cached_key)
                result.deleted += deleted
                self._hash_cache.remove(cached_key)
                logger.info("Removed vanished file from index: %s (%d chunks)", cached_key, deleted)

        self._hash_cache.save()
        result.finished_at = datetime.utcnow()
        result.duration_seconds = time.monotonic() - t0
        logger.info(
            "Index pass complete: scanned=%d indexed=%d skipped=%d deleted=%d chunks=%d errors=%d duration=%.2fs",
            result.scanned, result.indexed, result.skipped, result.deleted,
            result.chunks_created, len(result.errors), result.duration_seconds,
        )
        return result

    def _index_one(self, path: Path, result: IndexResult, seen_files: set[str]) -> None:
        from palace.utils.hashing import hash_file

        source_file = str(path)
        seen_files.add(source_file)

        try:
            file_hash = hash_file(path)
            mtime = path.stat().st_mtime
        except OSError as exc:
            result.errors.append(f"Cannot read {path}: {exc}")
            return

        if self._hash_cache.is_unchanged(source_file, file_hash):
            result.skipped += 1
            logger.debug("Skipping unchanged file: %s", source_file)
            return

        logger.info("Indexing changed/new file: %s", source_file)
        chunker = MarkdownASTChunker(
            self._config.chunker,
            self._extractor,
            source_hash_override=file_hash,
        )
        chunks = chunker.chunk_file(path)
        if not chunks:
            result.skipped += 1
            return

        # Batch embed all chunks for this file at once.
        texts = [c.content for c in chunks]
        t_embed = time.monotonic()
        vectors = self._embedder.embed_batch(texts)
        embed_ms = (time.monotonic() - t_embed) * 1000
        logger.debug("Embedded %d chunks in %.1fms", len(chunks), embed_ms)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        # Attach embeddings to each chunk.  ChunkMetadata is frozen, but the
        # embedding lives on the Chunk itself, so we rebuild each Chunk with
        # its vector while preserving the original metadata.
        populated: list[Chunk] = [
            Chunk(metadata=chunk.metadata, content=chunk.content, embedding=vec)
            for chunk, vec in zip(chunks, vectors)
        ]

        self._repository.update_chunks(source_file, populated)
        self._hash_cache.upsert(source_file, file_hash, mtime, len(populated))
        result.indexed += 1
        result.chunks_created += len(populated)

    # -- maintenance --

    def reindex_all(self) -> IndexResult:
        """Force re-indexing of every file by clearing the hash cache first."""
        self._hash_cache.load()
        for key in list(self._hash_cache.all_keys()):
            self._hash_cache.remove(key)
        self._hash_cache.save()
        return self.index()

    def verify(self) -> dict[str, int]:
        """Check repository contents against the hash cache.

        Re-hashes every cached file on disk and compares it to the stored
        content hash.  Does not modify the index — use :meth:`reindex_all`
        to repair drift.

        Returns counts of healthy, stale, and missing files plus the total
        chunk count in the repository.
        """
        from palace.utils.hashing import hash_file

        self._hash_cache.load()
        stats = self._repository.statistics()
        cache_keys = self._hash_cache.all_keys()
        healthy = 0
        stale = 0
        missing = 0
        for key in cache_keys:
            entry = self._hash_cache.get(key)
            if entry is None:
                continue
            path = Path(key)
            if not path.exists():
                missing += 1
                continue
            try:
                current_hash = hash_file(path)
            except OSError:
                missing += 1
                continue
            if current_hash == entry.content_hash:
                healthy += 1
            else:
                stale += 1
        return {
            "cached_files": len(cache_keys),
            "repository_chunks": stats.get("total_chunks", 0),
            "healthy_files": healthy,
            "stale_files": stale,
            "missing_files": missing,
        }


__all__ = ["IncrementalIndexer", "IndexResult"]
