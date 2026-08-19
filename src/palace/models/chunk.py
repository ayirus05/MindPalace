"""Core data models for the MindPalace Tier 2 semantic engine.

All domain objects are immutable Pydantic models. Downstream layers
(embeddings, indexing, search, repository) depend on these types, never
the other way around — this keeps the package cleanly layered.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentDomain(str, Enum):
    """High-level provenance of a document, used for filtering and weighting."""

    JOURNAL = "journal"
    NOTES = "notes"
    TRANSCRIPT = "transcript"
    CONVERSATION = "conversation"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | None) -> DocumentDomain:
        """Best-effort coercion; unknown values fall back to UNKNOWN."""
        if value is None:
            return cls.UNKNOWN
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk. Persisted alongside the vector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    source_file: str
    source_file_hash: str
    domain: DocumentDomain = DocumentDomain.UNKNOWN
    date: Optional[date]
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    content_hash: str
    word_count: int = 0
    token_estimate: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("word_count", "token_estimate")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("counts must be non-negative")
        return v


class Chunk(BaseModel):
    """A unit of text plus its embedding and metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: ChunkMetadata
    content: str
    embedding: list[float] = Field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def source_file(self) -> str:
        return self.metadata.source_file


class RawHit(BaseModel):
    """A single result from the vector store, before ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    content: str
    score: float
    source_file: str
    date: Optional[date]
    domain: DocumentDomain = DocumentDomain.UNKNOWN
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    word_count: int = 0


class SearchResult(BaseModel):
    """A final, ranked result returned to the caller (or future MCP layer)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    score: float
    source_file: str
    date: Optional[date]
    domain: DocumentDomain = DocumentDomain.UNKNOWN
    tags: list[str] = Field(default_factory=list)
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexStats(BaseModel):
    """Snapshot of index state for the stats command and diagnostics."""

    model_config = ConfigDict(extra="forbid")

    total_chunks: int = 0
    unique_source_files: int = 0
    by_domain: dict[str, int] = Field(default_factory=dict)
    by_extension: dict[str, int] = Field(default_factory=dict)
    date_range_earliest: Optional[date]
    date_range_latest: Optional[date]
    db_path: str = ""
    last_indexed_at: Optional[datetime]
