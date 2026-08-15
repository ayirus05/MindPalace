"""Shared YAKE keyword extraction for indexing and retrieval."""

from __future__ import annotations

import yake


_KEYWORD_EXTRACTOR = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5)


def extract_keywords(text: str) -> list[str]:
    """Return YAKE's highest-ranked keyword phrases for ``text``."""
    return [
        keyword
        for keyword, _score in _KEYWORD_EXTRACTOR.extract_keywords(text)
    ]


__all__ = ["extract_keywords"]
