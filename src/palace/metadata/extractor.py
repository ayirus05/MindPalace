"""Metadata extraction strategies.

Given a source file and (optionally) the text of a single entry, produce the
:class:`ChunkMetadata` fields that depend on content rather than structure:
``domain``, ``date``, and ``tags``.

Each strategy is a small class so they can be tested in isolation and the
order of precedence is explicit.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path

from palace.models.chunk import DocumentDomain
from palace.models.config import ChunkerConfig
from palace.utils.hashing import TextEstimator, sha256_text


logger = logging.getLogger("palace.metadata")


# ---- Date extraction ------------------------------------------------------

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_FULL = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}


def extract_date(text: str) -> date | None:
    """Find the first parseable date in ``text``.

    Tries ISO dates first (``2026-04-15``), then long-form
    (``April 15, 2026``), then slash dates (``15/4/2026``).
    Returns ``None`` if none found.
    """
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _LONG_DATE_RE.search(text)
    if m:
        month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        month = _MONTH_FULL.get(month_name)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    m = _SLASH_DATE_RE.search(text)
    if m:
        d1, d2, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Heuristic: if the first number > 12, it's day-first.
        if d1 > 12:
            day, month = d1, d2
        else:
            month, day = d1, d2  # assume M/D/Y (US default)
        try:
            return date(year, month, day)
        except ValueError:
            pass

    return None


# ---- Domain inference ------------------------------------------------------

_PATH_DOMAIN_MAP = {
    "journal": DocumentDomain.JOURNAL,
    "journals": DocumentDomain.JOURNAL,
    "note": DocumentDomain.NOTES,
    "notes": DocumentDomain.NOTES,
    "transcript": DocumentDomain.TRANSCRIPT,
    "transcripts": DocumentDomain.TRANSCRIPT,
    "conversation": DocumentDomain.CONVERSATION,
    "conversations": DocumentDomain.CONVERSATION,
}


def infer_domain(source_path: Path) -> DocumentDomain:
    """Infer the document domain from the source file's path components."""
    parts = [p.lower() for p in source_path.parts]
    for key, domain in _PATH_DOMAIN_MAP.items():
        if key in parts:
            return domain
    return DocumentDomain.UNKNOWN


# ---- Tag extraction (YAML front-matter + inline #tags) --------------------

_YAML_FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_LINE_RE = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9])")


def extract_tags(text: str) -> list[str]:
    """Extract tags from YAML front-matter ``tags:`` field and inline ``#tag`` syntax."""
    tags: list[str] = []
    seen: set[str] = set()

    fm = _YAML_FRONT_RE.search(text)
    if fm:
        for line in fm.group(1).splitlines():
            line_stripped = line.strip()
            if line_stripped.lower().startswith("tags:"):
                value = line_stripped.split(":", 1)[1].strip()
                for tag in re.findall(r"[A-Za-z0-9_-]+", value):
                    tag_l = tag.lower()
                    if tag_l not in seen:
                        seen.add(tag_l)
                        tags.append(tag_l)

    # Inline #tags outside the front-matter.
    body = text[fm.end():] if fm else text
    for m in _TAG_LINE_RE.finditer(body):
        tag = m.group(1).lower()
        if tag not in seen and not tag.isdigit():
            seen.add(tag)
            tags.append(tag)

    return tags


# ---- Composite metadata builder -------------------------------------------

class MetadataExtractor:
    """Composite extractor yielding the content-derived metadata for a chunk.

    Structural fields (``chunk_id``, ``created_at``, ``updated_at``) are
    populated by ``MarkdownASTChunker``, not here — this class is responsible
    only for fields derivable from the source path and chunk text.
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self._estimator = TextEstimator(tokens_per_word=config.tokens_per_word)

    def extract(
        self,
        source_path: Path,
        chunk_text: str,
        source_hash: str,
    ) -> tuple[DocumentDomain, date | None, list[str], int, int]:
        """Return (domain, date, tags, word_count, token_estimate)."""
        domain = infer_domain(source_path)
        d = extract_date(chunk_text) or extract_date(source_path.name) or None
        tags = extract_tags(chunk_text)
        wc = self._estimator.word_count(chunk_text)
        tok = self._estimator.token_estimate(chunk_text)
        return domain, d, tags, wc, tok

    def build_metadata(
        self,
        source_path: Path,
        chunk_text: str,
        source_hash: str,
    ) -> tuple[DocumentDomain, date | None, list[str], str, int, int]:
        """Full metadata extraction; returns fields needed for ChunkMetadata.

        Returns ``(domain, date, tags, content_hash, word_count, token_estimate)``.
        """
        domain, d, tags, wc, tok = self.extract(source_path, chunk_text, source_hash)
        content_hash = sha256_text(chunk_text)
        return domain, d, tags, content_hash, wc, tok


__all__ = [
    "MetadataExtractor",
    "extract_date",
    "infer_domain",
    "extract_tags",
]
