"""Validation models for search requests."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from palace.models.chunk import DocumentDomain


class SearchFilter(BaseModel):
    """Validated filters accepted by the semantic search engine."""

    model_config = ConfigDict(extra="forbid")

    domain: DocumentDomain | None = None
    date_from: date | None = None
    date_to: date | None = None
    tags: list[str] | None = None
    metadata_filters: dict[str, Any] | None = None
    top_k: int | None = None

    @field_validator("domain", mode="before")
    @classmethod
    def _coerce_domain(cls, value: Any) -> DocumentDomain | None:
        if value is None or isinstance(value, DocumentDomain):
            return value
        if isinstance(value, str):
            return DocumentDomain.from_string(value)
        return value

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _coerce_date(cls, value: Any) -> date | None:
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return value


__all__ = ["SearchFilter"]
