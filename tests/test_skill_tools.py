"""Tests for LLM-facing memory tools."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from palace.skills import tools


def test_append_to_journal_creates_and_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journals_dir = tmp_path / "data" / "source_docs" / "journals"
    monkeypatch.setattr(tools, "JOURNALS_DIR", journals_dir)

    first = tools.append_to_journal("2026-08-20", "First entry")
    second = tools.append_to_journal("2026-08-20", "Second entry")

    path = journals_dir / "2026-08-20.md"
    assert first == {"status": "success", "path": str(path)}
    assert second == {"status": "success", "path": str(path)}
    assert path.read_text(encoding="utf-8") == "- First entry\n- Second entry\n"


@pytest.mark.parametrize("date_str", [None, ""])
def test_append_to_journal_defaults_to_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, date_str: str | None
) -> None:
    journals_dir = tmp_path / "journals"
    monkeypatch.setattr(tools, "JOURNALS_DIR", journals_dir)

    result = tools.append_to_journal(date_str, "Today's entry")

    path = journals_dir / f"{date.today().isoformat()}.md"
    assert result == {"status": "success", "path": str(path)}
    assert path.read_text(encoding="utf-8") == "- Today's entry\n"


@pytest.mark.parametrize("date_str", ["today", "2026-02-30", "../2026-08-20"])
def test_append_to_journal_rejects_invalid_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, date_str: str
) -> None:
    monkeypatch.setattr(tools, "JOURNALS_DIR", tmp_path / "journals")

    with pytest.raises(ValueError, match="Invalid journal date format"):
        tools.append_to_journal(date_str, "Entry")
