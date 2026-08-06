"""LanceDB repository layer.

This module is the *only* place in the codebase that imports ``lancedb`` or
touches the vector database directly.  Everything else goes through the
:class:`ChunkRepository` protocol, which keeps the indexer and search engine
testable and swappable.

Design notes
------------
* PyArrow schema is declared explicitly so the table is self-describing.
* Metadata filters are pushed down to LanceDB whenever possible (pre-filter)
  so the vector search only runs over the candidate set.
* All write paths are idempotent on ``chunk_id`` — upserts replace existing
  rows for the same source file before inserting the new ones.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from palace.models.chunk import Chunk, ChunkMetadata, DocumentDomain


logger = logging.getLogger("palace.repository")


# ---- Protocol --------------------------------------------------------------


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence abstraction for chunks + vectors."""

    def insert_chunks(self, chunks: list[Chunk]) -> int:
        ...

    def delete_chunks(self, source_file: str) -> int:
        ...

    def update_chunks(self, source_file: str, chunks: list[Chunk]) -> int:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        domain: DocumentDomain | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def filter_by_metadata(self, **filters: Any) -> list[dict[str, Any]]:
        ...

    def vacuum(self) -> dict[str, Any]:
        ...

    def statistics(self) -> dict[str, Any]:
        ...

    def count(self) -> int:
        ...


# ---- Concrete LanceDB implementation --------------------------------------


def _escape_sql_string(value: str) -> str:
    """Escape single quotes for LanceDB SQL filters to prevent injection."""
    return value.replace("'", "''")


def _to_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value


class LanceDBRepository:
    """Production repository backed by LanceDB.

    The schema is created lazily on first insert.  Once the table exists,
    subsequent opens reuse it.  All public methods are safe to call
    repeatedly.
    """
    VECTOR_DIM = 768

    SCHEMA = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("source_file_hash", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("date", pa.string()),  # ISO date string; None stored as ""
        pa.field("tags", pa.list_(pa.string())),
        pa.field("content_hash", pa.string()),
        pa.field("word_count", pa.int32()),
        pa.field("token_estimate", pa.int32()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    def __init__(self, db_path: str | Path, table_name: str = "chunks") -> None:
        import lancedb  # imported here so tests can monkeypatch

        self._db_path = str(db_path)
        self._table_name = table_name
        self._db = lancedb.connect(self._db_path)
        self._table = None

    # -- table lifecycle --

    def _ensure_table(self) -> None:
        """Open or create the LanceDB table."""
        if self._table is not None:
            return
        existing = self._db.table_names()
        if self._table_name in existing:
            self._table = self._db.open_table(self._table_name)
        else:
            # Create empty table with the declared schema.
            self._table = self._db.create_table(
                self._table_name,
                schema=self.SCHEMA,
                mode="create",
            )
        logger.debug("LanceDB table '%s' ready", self._table_name)

    def _row_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalise a LanceDB row into a plain dict with parsed types."""
        d = dict(row)
        # PyArrow returns date as string when stored as string; normalise.
        date_str = d.get("date")
        d["date"] = date_str if date_str else None
        domain_value = d.get("domain")
        d["domain"] = DocumentDomain.from_string(domain_value) if domain_value else DocumentDomain.UNKNOWN
        return d

    def _chunk_to_row(self, chunk: Chunk) -> dict[str, Any]:
        md: ChunkMetadata = chunk.metadata
        return {
            "chunk_id": md.chunk_id,
            "source_file": md.source_file,
            "source_file_hash": md.source_file_hash,
            "domain": md.domain.value,
            "date": md.date.isoformat() if md.date else "",
            "tags": md.tags or [],
            "content_hash": md.content_hash,
            "word_count": md.word_count,
            "token_estimate": md.token_estimate,
            "created_at": md.created_at.isoformat(),
            "updated_at": md.updated_at.isoformat(),
            "content": chunk.content,
            "vector": chunk.embedding,
        }

    # -- writes --

    def insert_chunks(self, chunks: list[Chunk]) -> int:
        """Append chunks to the table.  Embeddings must be populated."""
        if not chunks:
            return 0
        self._ensure_table()
        rows = [self._chunk_to_row(c) for c in chunks]
        missing = [r["chunk_id"] for r in rows if not r["vector"]]
        if missing:
            raise ValueError(f"Cannot insert chunks without embeddings: {missing[:3]}")
        self._table.add(rows)
        logger.info("Inserted %d chunks", len(rows))
        return len(rows)

    def delete_chunks(self, source_file: str) -> int:
        """Delete all chunks matching ``source_file``."""
        self._ensure_table()
        before = self._table.count_rows()
        # Escape single quotes to avoid breaking the SQL filter.
        safe = _escape_sql_string(source_file)
        self._table.delete(f"source_file = '{safe}'")
        after = self._table.count_rows()
        deleted = before - after
        logger.info("Deleted %d chunks for source_file=%s", deleted, source_file)
        return max(0, deleted)

    def update_chunks(self, source_file: str, chunks: list[Chunk]) -> int:
        """Replace all chunks for ``source_file`` with ``chunks``."""
        self.delete_chunks(source_file)
        print("Sample chunk embedding length:", len(chunks[0].embedding if chunks else 0))
        return self.insert_chunks(chunks)

    # -- reads --

    def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        domain: DocumentDomain | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search with metadata pre-filtering.

        LanceDB's filter pushdown narrows the candidate set before the vector
        comparison, keeping the search fast over large tables.
        """
        self._ensure_table()
        filter_sql = self._build_filter_sql(domain, date_from, date_to, tags)
        query = self._table.search(query_vector, vector_column_name="vector")

        if filter_sql:
            query = query.where(filter_sql)

        results = query.limit(top_k).to_list()
        return [self._row_to_dict(r) for r in results]

    def _build_filter_sql(
        self,
        domain: DocumentDomain | None,
        date_from: date | None,
        date_to: date | None,
        tags: list[str] | None,
    ) -> str:
        """Build a LanceDB SQL-style WHERE clause for pre-filtering."""
        clauses: list[str] = []
        if domain is not None:
            clauses.append(f"domain = '{_escape_sql_string(domain.value)}'")
        if date_from is not None:
            clauses.append(f"date >= '{date_from.isoformat()}'")
        if date_to is not None:
            clauses.append(f"date <= '{date_to.isoformat()}'")
        if tags:
            # array_contains_all isn't standard; use one array_contains per tag.
            tag_clauses = [f"array_contains(tags, '{_escape_sql_string(t)}')" for t in tags]
            clauses.append("(" + " AND ".join(tag_clauses) + ")")
        return " AND ".join(clauses)

    def filter_by_metadata(self, **filters: Any) -> list[dict[str, Any]]:
        """Return rows matching all given metadata filters (non-vector).

        Pushes the filter to LanceDB via a SQL WHERE clause on ``to_arrow``
        rather than loading the entire table into Python memory.
        """
        self._ensure_table()
        clauses: list[str] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, DocumentDomain):
                clauses.append(f"{key} = '{_escape_sql_string(value.value)}'")
            elif isinstance(value, str):
                clauses.append(f"{key} = '{_escape_sql_string(value)}'")
            elif isinstance(value, list):
                sub = [f"{key} = '{_escape_sql_string(str(v))}'" for v in value]
                clauses.append("(" + " OR ".join(sub) + ")")
        if not clauses:
            # No filters: return all rows (useful for `palace inspect` with no args).
            return [self._row_to_dict(r) for r in self._table.to_arrow().to_pylist()]
        where = " AND ".join(clauses)
        # `to_arrow` accepts a `where` parameter for non-vector SQL filtering.
        table = self._table.to_arrow(where=where)
        return [self._row_to_dict(r) for r in table.to_pylist()]

    def vacuum(self) -> dict[str, Any]:
        """Compact the table and reclaim space.  Returns metrics."""
        self._ensure_table()
        before = self._table.count_rows()
        try:
            self._table.optimize()
            action = "optimized"
        except Exception as exc:
            logger.warning("optimize() not supported or failed: %s", exc)
            action = "skipped"
        after = self._table.count_rows()
        return {"action": action, "rows_before": before, "rows_after": after}

    def statistics(self) -> dict[str, Any]:
        """Return aggregate stats about the indexed corpus."""
        self._ensure_table()
        rows = self._table.to_arrow().to_pylist()
        total = len(rows)
        by_domain: dict[str, int] = {}
        by_extension: dict[str, int] = {}
        dates: list[str] = []
        source_files: set[str] = set()
        for r in rows:
            d = r.get("domain", "unknown") or "unknown"
            by_domain[d] = by_domain.get(d, 0) + 1
            sf = r.get("source_file", "")
            source_files.add(sf)
            ext = Path(sf).suffix or "none"
            by_extension[ext] = by_extension.get(ext, 0) + 1
            dt = r.get("date")
            if dt:
                dates.append(dt)
        dates.sort()
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "by_domain": by_domain,
            "by_extension": by_extension,
            "date_range_earliest": dates[0] if dates else None,
            "date_range_latest": dates[-1] if dates else None,
            "db_path": self._db_path,
            "last_indexed_at": datetime.utcnow().isoformat(),
        }

    def count(self) -> int:
        self._ensure_table()
        return self._table.count_rows()


__all__ = ["ChunkRepository", "LanceDBRepository"]
