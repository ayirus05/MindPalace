"""Tests for the chunker."""

from __future__ import annotations

from pathlib import Path

from palace.indexing.chunker import MarkdownASTChunker
from palace.metadata.extractor import MetadataExtractor
from palace.models.chunk import DocumentDomain
from palace.models.config import ChunkerConfig


class TestMarkdownASTChunker:
    @staticmethod
    def build_chunker(source_hash: str | None = None) -> MarkdownASTChunker:
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

        chunks = self.build_chunker().chunk_by_headers(markdown)

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

        chunks = self.build_chunker().chunk_by_headers(markdown)

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

    def test_chunk_text_builds_validated_chunks_with_keywords(self) -> None:
        source_path = Path("notes/project.md")
        chunks = self.build_chunker("fixedhash").chunk_text(
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

        chunks = self.build_chunker().chunk_file(path)

        assert [chunk.content for chunk in chunks] == ["Ship it."]
