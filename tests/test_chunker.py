"""Tests for the chunker."""

from __future__ import annotations

from pathlib import Path

from palace.indexing.chunker import (
    Chunker,
    MarkdownASTChunker,
    make_chunker,
    split_journal_entries,
)
from palace.metadata.extractor import MetadataExtractor
from palace.models.chunk import DocumentDomain
from palace.models.config import ChunkerConfig


class TestMarkdownASTChunker:
    @staticmethod
    def make_chunker(source_hash: str | None = None) -> MarkdownASTChunker:
        config = ChunkerConfig()
        return MarkdownASTChunker(
            config,
            MetadataExtractor(config),
            source_hash_override=source_hash,
        )

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

        chunks = self.make_chunker().chunk_by_headers(markdown)

        assert [
            {key: value for key, value in chunk.items() if key != "keywords"}
            for chunk in chunks
        ] == [
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
        assert all(chunk["keywords"] for chunk in chunks)
        assert all(
            isinstance(keyword, str)
            for chunk in chunks
            for keyword in chunk["keywords"]
        )

    def test_creates_one_chunk_per_header_including_empty_parent(self) -> None:
        markdown = """# Home Lab

Lab overview.

## Proxmox

### LXC Containers

Tailscale runs inside this container.

## Networking

Tailscale connects remote hosts.
"""

        chunks = self.make_chunker().chunk_by_headers(markdown)

        assert len(chunks) == 4
        assert [chunk["context_path"] for chunk in chunks] == [
            "Home Lab",
            "Home Lab > Proxmox",
            "Home Lab > Proxmox > LXC Containers",
            "Home Lab > Networking",
        ]
        assert chunks[1]["text"] == ""
        assert "Tailscale" not in chunks[0]["embed_text"]
        assert "Tailscale" in chunks[2]["embed_text"]

        proxmox_keywords = " ".join(chunks[1]["keywords"]).casefold()
        tailscale_keywords = " ".join(chunks[2]["keywords"]).casefold()
        assert "proxmox" in proxmox_keywords
        assert "tailscale" in tailscale_keywords

    def test_implements_chunker_protocol(self) -> None:
        assert isinstance(self.make_chunker(), Chunker)

    def test_chunk_text_builds_validated_chunks_with_keywords(self) -> None:
        source_path = Path("notes/project.md")
        chunks = self.make_chunker("fixedhash").chunk_text(
            "# Home Lab\n\nBuild the Proxmox server.", source_path
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content == "Build the Proxmox server."
        assert chunk.metadata.source_file == str(source_path)
        assert chunk.metadata.source_file_hash == "fixedhash"
        assert chunk.metadata.domain == DocumentDomain.NOTES
        assert chunk.metadata.keywords

    def test_chunk_file_reads_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "notes" / "project.md"
        path.parent.mkdir()
        path.write_text("# Project\n\nShip it.", encoding="utf-8")

        chunks = self.make_chunker().chunk_file(path)

        assert [chunk.content for chunk in chunks] == ["Ship it."]


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


class TestMakeChunker:
    def test_journal_path(self, tmp_path: Path) -> None:
        path = tmp_path / "journals" / "x.md"
        extractor = MetadataExtractor(ChunkerConfig())
        chunker = make_chunker(path, ChunkerConfig(), extractor)
        assert isinstance(chunker, MarkdownASTChunker)

    def test_notes_path(self, tmp_path: Path) -> None:
        path = tmp_path / "notes" / "x.md"
        extractor = MetadataExtractor(ChunkerConfig())
        chunker = make_chunker(path, ChunkerConfig(), extractor)
        assert isinstance(chunker, MarkdownASTChunker)
