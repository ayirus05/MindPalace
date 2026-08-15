"""LanceDB repository layer.

This module is the *only* place in the codebase that imports ``lancedb`` or
touches the vector database directly. Everything else uses
``LanceDBRepository``.

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
from typing import Any

import pyarrow as pa

from palace.models.chunk import Chunk, ChunkMetadata, DocumentDomain


logger = logging.getLogger("palace.repository")


def _escape_sql_string(value: str) -> str:
    """Escape single quotes for LanceDB SQL filters to prevent injection."""
    return value.replace("'", "''")

class LanceDBRepository:
    """Production repository backed by LanceDB.

    The schema is created lazily on first insert.  Once the table exists,
    subsequent opens reuse it.  All public methods are safe to call
    repeatedly.
    """
    BASE_SCHEMA_FIELDS = [
        pa.field("chunk_id", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("source_file_hash", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("date", pa.string()),  # ISO date string; None stored as ""
        pa.field("tags", pa.list_(pa.string())),
        pa.field("keywords", pa.list_(pa.string())),
        pa.field("content_hash", pa.string()),
        pa.field("word_count", pa.int32()),
        pa.field("token_estimate", pa.int32()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("content", pa.string()),
    ]

    VECTOR_FIELD_NAME = "vector"

    @classmethod
    def schema_for_dim(cls, dim: int) -> pa.Schema:
        return pa.schema([
            *cls.BASE_SCHEMA_FIELDS,
            pa.field(cls.VECTOR_FIELD_NAME, pa.list_(pa.float32(), dim)),
        ])

    def __init__(self, db_path: str | Path, table_name: str = "chunks") -> None:
        import lancedb  # imported here so tests can monkeypatch

        self._db_path = str(db_path)
        self._table_name = table_name
        self._db = lancedb.connect(self._db_path)
        self._table = None

    # -- table lifecycle --

    def _ensure_table(self, vector_dim: int | None = None) -> None:
        """Open or create the LanceDB table."""
        if self._table is not None:
            return
        existing = self._db.table_names()
        if self._table_name in existing:
            self._table = self._db.open_table(self._table_name)
            return
        if vector_dim is None:
            raise ValueError("Cannot create LanceDB table without vector dimensionality")
        self._table = self._db.create_table(
            self._table_name,
            schema=self.schema_for_dim(vector_dim),
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
            "keywords": md.keywords or [],
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
        rows = [self._chunk_to_row(c) for c in chunks]
        missing = [r["chunk_id"] for r in rows if not r["vector"]]
        if missing:
            raise ValueError(f"Cannot insert chunks without embeddings: {missing[:3]}")
        dims = [len(r["vector"]) for r in rows]
        if any(d == 0 for d in dims):
            raise ValueError("Cannot insert chunks with empty embedding vectors")
        if len(set(dims)) != 1:
            raise ValueError("All chunk embeddings must have the same dimensionality")
        self._ensure_table(vector_dim=dims[0])
        self._table.add(rows)
        logger.info("Inserted %d chunks", len(rows))
        return len(rows)

    def delete_chunks(self, source_file: str) -> int:
        """Delete all chunks matching ``source_file``."""
        if self._table is None and self._table_name not in self._db.table_names():
            return 0
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
        if self._table is None and self._table_name not in self._db.table_names():
            return []
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
        """Return rows matching all given metadata filters (non-vector)."""
        if self._table is None and self._table_name not in self._db.table_names():
            return []
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
            return [self._row_to_dict(r) for r in self._table.to_arrow().to_pylist()]
        where = " AND ".join(clauses)
        rows = [self._row_to_dict(r) for r in self._table.to_arrow().to_pylist()]
        def match_row(row: dict[str, Any]) -> bool:
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    if row.get(key) not in value:
                        return False
                else:
                    if row.get(key) != value and (not isinstance(value, DocumentDomain) or row.get(key) != value.value):
                        return False
            return True
        return [row for row in rows if match_row(row)]

    def vacuum(self) -> dict[str, Any]:
        """Compact the table and reclaim space.  Returns metrics."""
        if self._table is None and self._table_name not in self._db.table_names():
            return {"action": "skipped", "rows_before": 0, "rows_after": 0}
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
        if self._table is None and self._table_name not in self._db.table_names():
            return {
                "total_chunks": 0,
                "unique_source_files": 0,
                "by_domain": {},
                "by_extension": {},
                "date_range_earliest": None,
                "date_range_latest": None,
                "db_path": self._db_path,
                "last_indexed_at": datetime.utcnow().isoformat(),
            }
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
        if self._table is None and self._table_name not in self._db.table_names():
            return 0
        self._ensure_table()
        return self._table.count_rows()


__all__ = ["LanceDBRepository"]
