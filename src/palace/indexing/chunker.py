"""Chunking strategies for journals and Markdown notes.

The :class:`Chunker` protocol keeps the indexer decoupled from the concrete
splitting logic.  :class:`JournalChunker` recognises date markers and emits
one chunk per dated entry; :class:`NoteChunker` uses a sliding word window
to hit a target token count with configurable overlap.  For callers that
embed Markdown directly, :class:`MarkdownASTChunker` groups content by its
active header hierarchy.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from markdown_it import MarkdownIt
import yake

from palace.metadata.extractor import MetadataExtractor, extract_date
from palace.models.chunk import Chunk, ChunkMetadata, DocumentDomain
from palace.models.config import ChunkerConfig
from palace.utils.hashing import TextEstimator, sha256_text


logger = logging.getLogger("palace.chunker")

_KEYWORD_EXTRACTOR = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5)


def extract_keywords(embed_text: str) -> list[str]:
    """Extract ranked keyword strings from text prepared for embedding."""
    return [
        keyword
        for keyword, _score in _KEYWORD_EXTRACTOR.extract_keywords(embed_text)
    ]


class MarkdownASTChunker:
    """Split Markdown into semantic sections based on heading hierarchy.

    Heading text is carried in ``context_path`` rather than repeated in the
    section body.  The body itself is sliced from the source so Markdown
    formatting (including lists, links, and fenced code) remains intact.
    """

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark")

    def chunk_by_headers(self, markdown_text: str) -> list[dict]:
        """Return one embedding payload for each non-empty Markdown section."""
        tokens = self._markdown.parse(markdown_text)
        lines = markdown_text.splitlines(keepends=True)
        active_headers: dict[int, str] = {}
        chunks: list[dict] = []
        section_start = 0
        context_path = ""

        def append_section(end_line: int) -> None:
            text = "".join(lines[section_start:end_line]).strip()
            if not text:
                return
            embed_text = f"Context: {context_path}\n\n{text}"
            chunks.append(
                {
                    "context_path": context_path,
                    "text": text,
                    "embed_text": embed_text,
                    "keywords": extract_keywords(embed_text),
                }
            )

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type != "heading_open":
                index += 1
                continue

            # Block heading tokens always carry their source line range under
            # CommonMark.  Guarding the map keeps the chunker safe if plugins
            # introduce synthetic headings later.
            if token.map is None:
                index += 1
                continue

            append_section(token.map[0])

            level = int(token.tag.removeprefix("h"))
            for existing_level in list(active_headers):
                if existing_level >= level:
                    del active_headers[existing_level]

            heading_text = ""
            index += 1
            while index < len(tokens) and tokens[index].type != "heading_close":
                if tokens[index].type == "inline":
                    heading_text = tokens[index].content.strip()
                index += 1

            active_headers[level] = heading_text
            context_path = " > ".join(
                active_headers[header_level]
                for header_level in sorted(active_headers)
            )
            section_start = token.map[1]
            index += 1

        append_section(len(lines))
        return chunks


@runtime_checkable
class Chunker(Protocol):
    """Chunk a single source file into :class:`Chunk` instances."""

    def chunk_file(self, path: Path) -> list[Chunk]:
        ...

    def chunk_text(self, text: str, source_path: Path) -> list[Chunk]:
        ...


# ---- Journal entry splitter ------------------------------------------------

# A journal entry is assumed to begin with a date line.  We accept ISO dates
# (``2026-04-15``), long-form (``April 15, 2026``), and slash dates.  An entry
# runs until the next recognised date header or EOF.
_DATE_HEADER_RE = re.compile(
    r"^(?:(\d{4}-\d{2}-\d{2})"                                   # ISO
    r"|((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})"
    r"|(\d{1,2}/\d{1,2}/\d{4}))\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_journal_entries(text: str) -> list[tuple[str, str]]:
    """Split a journal file into ``(header, body)`` tuples.

    Returns one tuple per dated entry.  Leading content before the first
    dated header (e.g. a YAML front-matter block) is attached as preamble
    to the first entry, or returned as a single front-matter-only entry.
    """
    matches = list(_DATE_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    entries: list[tuple[str, str]] = []
    # Preamble before the first header: keep as a separate chunk if substantial.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        entries.append(("preamble", preamble))

    for i, m in enumerate(matches):
        head = m.group(0).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            entries.append((head, body))
    return entries


# ---- Concrete chunkers -----------------------------------------------------


class JournalChunker:
    """Chunker for journal files: one chunk per dated entry.

    Each entry may correspond to a day; entry text is kept intact (not
    re-split) to preserve narrative coherence.  If an entry is extremely
    long, it is further split using the token-budget splitter.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        extractor: MetadataExtractor,
        source_hash_override: str | None = None,
    ) -> None:
        self._config = config
        self._extractor = extractor
        self._estimator = TextEstimator(tokens_per_word=config.tokens_per_word)
        self._source_hash_override = source_hash_override

    def chunk_file(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, path)

    def chunk_text(self, text: str, source_path: Path) -> list[Chunk]:
        source_hash = self._source_hash_override or sha256_file_safe(source_path, text)
        entries = split_journal_entries(text)
        chunks: list[Chunk] = []
        for header, body in entries:
            # Strip front-matter-only preambles (handled in extractor tags).
            if header == "preamble" and body.startswith("---"):
                continue
            body = body.strip()
            if not body:
                continue
            entry_date = extract_date(header)
            body_hash = sha256_text(body)
            target = self._config.target_tokens
            if self._estimator.token_estimate(body) <= target * 2:
                chunks.append(self._build_chunk(source_path, body, source_hash, entry_date))
            else:
                # Long entry: split into token-budgeted sub-chunks.
                for segment in self._estimator.split_to_token_budget(body, target):
                    chunks.append(self._build_chunk(source_path, segment, source_hash, entry_date))
        if not chunks:
            logger.debug("No chunks produced from %s", source_path)
        return chunks

    def _build_chunk(
        self,
        source_path: Path,
        text: str,
        source_hash: str,
        explicit_date: date | None = None,
    ) -> Chunk:
        domain, d, tags, content_hash, wc, tok = self._extractor.build_metadata(
            source_path, text, source_hash
        )
        if explicit_date is not None:
            d = explicit_date
        metadata = ChunkMetadata(
            source_file=str(source_path),
            source_file_hash=source_hash,
            domain=domain,
            date=d,
            tags=tags,
            content_hash=content_hash,
            word_count=wc,
            token_estimate=tok,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return Chunk(metadata=metadata, content=text)


class NoteChunker:
    """Chunker for general notes: sliding word window with overlap.

    Honours ``target_tokens`` and ``overlap_tokens`` from the configuration.
    YAML front-matter is preserved as part of the first chunk so metadata
    extraction still works.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        extractor: MetadataExtractor,
        source_hash_override: str | None = None,
    ) -> None:
        self._config = config
        self._extractor = extractor
        self._estimator = TextEstimator(tokens_per_word=config.tokens_per_word)
        self._source_hash_override = source_hash_override

    def chunk_file(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, path)

    def chunk_text(self, text: str, source_path: Path) -> list[Chunk]:
        source_hash = self._source_hash_override or sha256_file_safe(source_path, text)

        # Preserve front-matter on the first chunk for tag extraction.
        fm_match = re.match(r"\A---\s*\n.*?\n---\s*\n", text, re.DOTALL)
        front_matter = fm_match.group(0) if fm_match else ""
        body = text[fm_match.end():] if fm_match else text

        words = body.split()
        if not words:
            return []

        target_tokens = max(1, self._config.target_tokens)
        # Convert token budget to word budget.
        target_words = max(1, int(target_tokens / self._config.tokens_per_word))
        overlap_words = max(0, int(self._config.overlap_tokens / self._config.tokens_per_word))
        step = max(1, target_words - overlap_words)

        chunks: list[Chunk] = []
        i = 0
        first = True
        while i < len(words):
            window = words[i : i + target_words]
            segment_text = (" ".join(window)).strip()
            if segment_text:
                full_text = f"{front_matter}{segment_text}" if first else segment_text
                chunks.append(self._build_chunk(source_path, full_text, source_hash))
                first = False
            if i + target_words >= len(words):
                break
            i += step
        return chunks

    def _build_chunk(self, source_path: Path, text: str, source_hash: str) -> Chunk:
        domain, d, tags, content_hash, wc, tok = self._extractor.build_metadata(
            source_path, text, source_hash
        )
        metadata = ChunkMetadata(
            source_file=str(source_path),
            source_file_hash=source_hash,
            domain=domain,
            date=d,
            tags=tags,
            content_hash=content_hash,
            word_count=wc,
            token_estimate=tok,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return Chunk(metadata=metadata, content=text)


# ---- Factory --------------------------------------------------------------

def sha256_file_safe(path: Path, text: str) -> str:
    """Hash a file's bytes; fall back to hashing the in-memory text."""
    from palace.utils.hashing import hash_file, sha256_text

    try:
        return hash_file(path)
    except OSError:
        return sha256_text(text)


def make_chunker(
    source_path: Path,
    config: ChunkerConfig,
    extractor: MetadataExtractor,
    source_hash: str | None = None,
) -> Chunker:
    """Pick the right chunker for ``source_path`` based on its domain."""
    from palace.metadata.extractor import infer_domain

    domain = infer_domain(source_path)
    if domain == DocumentDomain.JOURNAL:
        return JournalChunker(config, extractor, source_hash_override=source_hash)
    return NoteChunker(config, extractor, source_hash_override=source_hash)


__all__ = [
    "Chunker",
    "JournalChunker",
    "MarkdownASTChunker",
    "NoteChunker",
    "extract_keywords",
    "make_chunker",
    "split_journal_entries",
    "sha256_file_safe",
]
