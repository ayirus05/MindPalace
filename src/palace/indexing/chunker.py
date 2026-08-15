"""Semantic Markdown chunking based on heading hierarchy."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from markdown_it import MarkdownIt

from palace.metadata.keywords import extract_keywords
from palace.metadata.extractor import MetadataExtractor
from palace.models.chunk import Chunk, ChunkMetadata
from palace.models.config import ChunkerConfig


class MarkdownASTChunker:
    """Split Markdown into semantic sections based on heading hierarchy.

    Heading text is carried in ``context_path`` rather than repeated in the
    section body.  The body itself is sliced from the source so Markdown
    formatting (including lists, links, and fenced code) remains intact.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        extractor: MetadataExtractor,
        source_hash_override: str | None = None,
    ) -> None:
        self._config = config
        self._extractor = extractor
        self._source_hash_override = source_hash_override
        self._markdown = MarkdownIt("commonmark")

    def chunk_file(self, path: Path) -> list[Chunk]:
        """Read and chunk a UTF-8 Markdown file."""
        return self.chunk_text(path.read_text(encoding="utf-8"), path)

    def chunk_text(self, text: str, source_path: Path) -> list[Chunk]:
        """Convert heading-delimited sections into validated chunks."""
        source_hash = self._source_hash_override or sha256_file_safe(source_path, text)
        chunks: list[Chunk] = []
        for section in self.chunk_by_headers(text):
            section_text = section["text"]
            domain, d, tags, content_hash, wc, tok = self._extractor.build_metadata(
                source_path, section_text, source_hash
            )
            metadata = ChunkMetadata(
                source_file=str(source_path),
                source_file_hash=source_hash,
                domain=domain,
                date=d,
                tags=tags,
                keywords=section["keywords"],
                content_hash=content_hash,
                word_count=wc,
                token_estimate=tok,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            chunks.append(Chunk(metadata=metadata, content=section_text))
        return chunks

    def chunk_by_headers(self, markdown_text: str) -> list[dict]:
        """Return one embedding payload per heading-delimited section."""
        tokens = self._markdown.parse(markdown_text)
        lines = markdown_text.splitlines(keepends=True)
        active_headers: dict[int, str] = {}
        chunks: list[dict] = []

        headings: list[tuple[int, int, int, str]] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type != "heading_open" or token.map is None:
                index += 1
                continue

            level = int(token.tag.removeprefix("h"))
            heading_text = ""
            heading_end = token.map[1]
            index += 1
            while index < len(tokens) and tokens[index].type != "heading_close":
                if tokens[index].type == "inline":
                    heading_text = tokens[index].content.strip()
                index += 1
            headings.append((token.map[0], heading_end, level, heading_text))
            index += 1

        if not headings:
            text = markdown_text.strip()
            if not text:
                return []
            embed_text = f"Context: \n\n{text}"
            return [
                {
                    "context_path": "",
                    "text": text,
                    "embed_text": embed_text,
                    "keywords": extract_keywords(embed_text),
                }
            ]

        for heading_index, (_start, body_start, level, heading_text) in enumerate(headings):
            body_end = (
                headings[heading_index + 1][0]
                if heading_index + 1 < len(headings)
                else len(lines)
            )

            for existing_level in list(active_headers):
                if existing_level >= level:
                    del active_headers[existing_level]
            active_headers[level] = heading_text
            context_path = " > ".join(
                active_headers[header_level]
                for header_level in sorted(active_headers)
            )

            # Keep every heading as its own chunk, including headings with no
            # direct body before a nested child. The context still gives YAKE
            # useful section-specific nouns in that case.
            text = "".join(lines[body_start:body_end]).strip()
            embed_text = f"Context: {context_path}\n\n{text}"
            chunks.append(
                {
                    "context_path": context_path,
                    "text": text,
                    "embed_text": embed_text,
                    "keywords": extract_keywords(embed_text),
                }
            )
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
    """Build the semantic Markdown chunker used for every document domain."""
    return MarkdownASTChunker(config, extractor, source_hash_override=source_hash)


__all__ = [
    "Chunker",
    "MarkdownASTChunker",
    "extract_keywords",
    "make_chunker",
    "split_journal_entries",
    "sha256_file_safe",
]
