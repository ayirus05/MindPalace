"""Tests for metadata extraction."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from palace.metadata.extractor import (
    MetadataExtractor,
    extract_date,
    extract_tags,
    infer_domain,
)
from palace.models.chunk import DocumentDomain
from palace.models.config import ChunkerConfig


class TestExtractDate:
    def test_iso_date(self) -> None:
        assert extract_date("2026-04-15") == date(2026, 4, 15)

    def test_iso_date_in_text(self) -> None:
        assert extract_date("Entry on 2026-04-15 was fine") == date(2026, 4, 15)

    def test_long_date(self) -> None:
        assert extract_date("April 15, 2026") == date(2026, 4, 15)

    def test_long_date_no_comma(self) -> None:
        assert extract_date("April 15 2026") == date(2026, 4, 15)

    def test_slash_date_us(self) -> None:
        # 4/15/2026 -> April 15 (US default, first <= 12)
        assert extract_date("4/15/2026") == date(2026, 4, 15)

    def test_slash_date_day_first(self) -> None:
        # 15/4/2026 -> first > 12 so day-first
        assert extract_date("15/4/2026") == date(2026, 4, 15)

    def test_no_date(self) -> None:
        assert extract_date("no date here") is None

    def test_multiple_dates_returns_first(self) -> None:
        assert extract_date("2026-01-01 then 2026-12-31") == date(2026, 1, 1)

    def test_invalid_date_falls_through(self) -> None:
        # 2026-13-40 is not a real date; should not match.
        assert extract_date("2026-13-40") is None


class TestInferDomain:
    def test_journal_path(self, tmp_path: Path) -> None:
        assert infer_domain(Path("/data/journals/2026.md")) == DocumentDomain.JOURNAL

    def test_notes_path(self, tmp_path: Path) -> None:
        assert infer_domain(Path("/data/notes/research.md")) == DocumentDomain.NOTES

    def test_transcript_path(self) -> None:
        assert infer_domain(Path("transcripts/meeting.md")) == DocumentDomain.TRANSCRIPT

    def test_unknown_path(self) -> None:
        assert infer_domain(Path("/random/file.md")) == DocumentDomain.UNKNOWN


class TestExtractTags:
    def test_yaml_front_matter_tags(self) -> None:
        text = "---\ntags: [health, energy]\n---\nbody"
        assert extract_tags(text) == ["health", "energy"]

    def test_inline_tags(self) -> None:
        text = "Feeling great today #health #exercise"
        tags = extract_tags(text)
        assert "health" in tags
        assert "exercise" in tags

    def test_combined_front_and_inline(self) -> None:
        text = "---\ntags: [health]\n---\nFeeling great #energy"
        tags = extract_tags(text)
        assert "health" in tags
        assert "energy" in tags

    def test_no_tags(self) -> None:
        assert extract_tags("just some text") == []

    def test_no_duplicate_tags(self) -> None:
        text = "---\ntags: [health]\n---\n#health again"
        assert extract_tags(text) == ["health"]

    def test_tag_with_hyphen(self) -> None:
        text = "tagged #sleep-quality"
        assert "sleep-quality" in extract_tags(text)

    def test_pure_numeric_tag_ignored(self) -> None:
        text = "#123"
        assert "123" not in extract_tags(text)


class TestMetadataExtractor:
    def test_build_metadata(self, tmp_path: Path) -> None:
        extractor = MetadataExtractor(ChunkerConfig())
        path = Path("/data/journals/2026-04-15.md")
        text = "2026-04-15\n\nFelt tired today."
        domain, d, tags, content_hash, wc, tok = extractor.build_metadata(path, text, "abc123")
        assert domain == DocumentDomain.JOURNAL
        assert d == date(2026, 4, 15)
        assert tags == []
        assert wc == 4  # "Felt tired today." splits to 3 words + date? let's check
        assert tok > 0
        assert len(content_hash) == 64  # sha256 hex

    def test_build_metadata_with_tags(self, tmp_path: Path) -> None:
        extractor = MetadataExtractor(ChunkerConfig())
        path = Path("/data/notes/research.md")
        text = "---\ntags: [finance, planning]\n---\n# rebalancing"
        domain, d, tags, content_hash, wc, tok = extractor.build_metadata(path, text, "hash")
        assert domain == DocumentDomain.NOTES
        assert "finance" in tags
        assert "planning" in tags
