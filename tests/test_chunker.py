"""Tests for the chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from palace.indexing.chunker import (
    JournalChunker,
    MarkdownASTChunker,
    NoteChunker,
    make_chunker,
    split_journal_entries,
)
from palace.metadata.extractor import MetadataExtractor
from palace.models.chunk import DocumentDomain
from palace.models.config import ChunkerConfig


class TestMarkdownASTChunker:
    def test_chunks_by_header_hierarchy(self) -> None:
        markdown = """# 2026 Goals

Plan for the year.

## Home Lab

Build the new server.

### Networking

- Configure VLANs
- Add firewall rules

## Fitness

Run three times a week.
"""

        chunks = MarkdownASTChunker().chunk_by_headers(markdown)

        assert chunks == [
            {
                "context_path": "2026 Goals",
                "text": "Plan for the year.",
                "embed_text": "Context: 2026 Goals\n\nPlan for the year.",
            },
            {
                "context_path": "2026 Goals > Home Lab",
                "text": "Build the new server.",
                "embed_text": (
                    "Context: 2026 Goals > Home Lab\n\nBuild the new server."
                ),
            },
            {
                "context_path": "2026 Goals > Home Lab > Networking",
                "text": "- Configure VLANs\n- Add firewall rules",
                "embed_text": (
                    "Context: 2026 Goals > Home Lab > Networking\n\n"
                    "- Configure VLANs\n- Add firewall rules"
                ),
            },
            {
                "context_path": "2026 Goals > Fitness",
                "text": "Run three times a week.",
                "embed_text": (
                    "Context: 2026 Goals > Fitness\n\nRun three times a week."
                ),
            },
        ]


class TestSplitJournalEntries:
    def test_multiple_entries(self) -> None:
        text = "2026-04-10\n\nFirst entry.\n\n2026-04-15\n\nSecond entry.\n"
        entries = split_journal_entries(text)
        assert len(entries) == 2
        assert "First entry" in entries[0][1]
        assert "Second entry" in entries[1][1]

    def test_preamble_before_first_date(self) -> None:
        text = "---\ntags: [x]\n---\n\n2026-04-10\n\nBody."
        entries = split_journal_entries(text)
        # Preamble (front-matter) + one dated entry.
        assert len(entries) == 2
        assert entries[0][0] == "preamble"

    def test_no_dates(self) -> None:
        text = "Just some text without dates."
        entries = split_journal_entries(text)
        assert len(entries) == 1
        assert entries[0][1] == text

    def test_long_form_date_header(self) -> None:
        text = "April 15, 2026\n\nLong form date entry.\n"
        entries = split_journal_entries(text)
        assert len(entries) == 1
        assert "Long form date entry" in entries[0][1]


class TestJournalChunker:
    def test_chunk_file(self, sample_journal: Path) -> None:
        config = ChunkerConfig()
        extractor = MetadataExtractor(config)
        chunker = JournalChunker(config, extractor)
        chunks = chunker.chunk_file(sample_journal)
        assert len(chunks) == 3  # one per dated entry
        # All chunks should have the journal domain.
        for c in chunks:
            assert c.metadata.domain == DocumentDomain.JOURNAL
        # Dates should be parsed.
        dates = [c.metadata.date for c in chunks if c.metadata.date]
        assert len(dates) >= 2

    def test_chunks_carry_source_hash(self, sample_journal: Path) -> None:
        config = ChunkerConfig()
        extractor = MetadataExtractor(config)
        chunker = JournalChunker(config, extractor, source_hash_override="fixedhash")
        chunks = chunker.chunk_file(sample_journal)
        for c in chunks:
            assert c.metadata.source_file_hash == "fixedhash"

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        config = ChunkerConfig()
        extractor = MetadataExtractor(config)
        chunker = JournalChunker(config, extractor)
        assert chunker.chunk_file(p) == []

    def test_long_entry_split(self, tmp_path: Path) -> None:
        """An entry whose token estimate exceeds 2x target gets sub-split."""
        p = tmp_path / "journals" / "long.md"
        p.parent.mkdir(parents=True)
        p.write_text(
            "2026-04-10\n\n" + ("word " * 1000),
            encoding="utf-8",
        )
        config = ChunkerConfig(target_tokens=50, tokens_per_word=1.0)
        extractor = MetadataExtractor(config)
        chunker = JournalChunker(config, extractor)
        chunks = chunker.chunk_file(p)
        assert len(chunks) > 1


class TestNoteChunker:
    def test_chunk_file_short(self, sample_note: Path) -> None:
        config = ChunkerConfig(target_tokens=300, overlap_tokens=40)
        extractor = MetadataExtractor(config)
        chunker = NoteChunker(config, extractor)
        chunks = chunker.chunk_file(sample_note)
        # Short note should produce one chunk.
        assert len(chunks) == 1
        assert "finance" in sample_note.read_text()
        # Front-matter tags should be extracted.
        assert "finance" in chunks[0].metadata.tags

    def test_chunk_long_note(self, tmp_path: Path) -> None:
        p = tmp_path / "notes" / "long.md"
        p.parent.mkdir(parents=True)
        p.write_text(" ".join(f"word{i}" for i in range(500)), encoding="utf-8")
        config = ChunkerConfig(target_tokens=50, overlap_tokens=10, tokens_per_word=1.0)
        extractor = MetadataExtractor(config)
        chunker = NoteChunker(config, extractor)
        chunks = chunker.chunk_file(p)
        assert len(chunks) > 1
        # Overlap means consecutive chunks share words.
        assert len(chunks) >= 8

    def test_source_hash_override(self, sample_note: Path) -> None:
        config = ChunkerConfig()
        extractor = MetadataExtractor(config)
        chunker = NoteChunker(config, extractor, source_hash_override="override")
        chunks = chunker.chunk_file(sample_note)
        assert all(c.metadata.source_file_hash == "override" for c in chunks)

    def test_front_matter_preserved_on_first_chunk(self, tmp_path: Path) -> None:
        p = tmp_path / "notes" / "fm.md"
        p.parent.mkdir(parents=True)
        body = " ".join(f"word{i}" for i in range(100))
        p.write_text(f"---\ntags: [alpha]\n---\n{body}", encoding="utf-8")
        config = ChunkerConfig(target_tokens=50, tokens_per_word=1.0)
        extractor = MetadataExtractor(config)
        chunker = NoteChunker(config, extractor)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1
        assert "alpha" in chunks[0].metadata.tags


class TestMakeChunker:
    def test_journal_path(self, tmp_path: Path) -> None:
        path = tmp_path / "journals" / "x.md"
        extractor = MetadataExtractor(ChunkerConfig())
        chunker = make_chunker(path, ChunkerConfig(), extractor)
        assert isinstance(chunker, JournalChunker)

    def test_notes_path(self, tmp_path: Path) -> None:
        path = tmp_path / "notes" / "x.md"
        extractor = MetadataExtractor(ChunkerConfig())
        chunker = make_chunker(path, ChunkerConfig(), extractor)
        assert isinstance(chunker, NoteChunker)
