"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force the FakeEmbedder for all tests unless explicitly overridden.
os.environ.setdefault("PALACE_EMBEDDER", "fake")

from palace.embeddings.manager import FakeEmbedder
from palace.indexing.hash_cache import HashCache
from palace.indexing.repository import LanceDBRepository
from palace.models.config import PalaceConfig


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A temporary project root with the standard subdirectories."""
    (tmp_path / "journals").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "index").mkdir()
    (tmp_path / "notes").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_project: Path) -> PalaceConfig:
    """A PalaceConfig rooted at the temp project."""
    cfg = PalaceConfig.default_for(tmp_project)
    cfg.indexer.source_dirs = ["journals"]
    cfg.database.path = "index/vector.lancedb"
    cfg.indexer.hash_cache_path = "cache/hash_cache.json"
    return cfg


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def repository(config: PalaceConfig) -> LanceDBRepository:
    return LanceDBRepository(
        db_path=config.resolve(config.database.path),
        table_name=config.database.table_name,
    )


@pytest.fixture
def hash_cache(config: PalaceConfig) -> HashCache:
    return HashCache(config.resolve(config.indexer.hash_cache_path))


@pytest.fixture
def sample_journal(tmp_project: Path) -> Path:
    """A sample multi-entry journal file."""
    p = tmp_project / "journals" / "2026-04.md"
    p.write_text(
        """---
tags: [health, energy]
---
2026-04-10

Started a new supplement protocol today. Feeling cautiously optimistic
about the new stack — magnesium, vitamin D, omega-3. Sleep was decent last
night, about 7 hours.

2026-04-15

Energy has been notably low this week. Waking up tired despite 7-8 hours
of sleep. Wondering if the new supplements are affecting my sleep quality.
Going to track sleep more carefully next week.

2026-04-22

Sleep metrics dipped in the second week — HRV down, deep sleep down.
Energy still low. I think there's a pattern here linking the supplement
change to fatigue. Will pause the evening dose and see.
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_note(tmp_project: Path) -> Path:
    """A general (non-journal) note file with front-matter tags."""
    p = tmp_project / "notes" / "research.md"
    p.write_text(
        """---
tags: [finance, planning]
---
# Financial planning notes

Quarterly review of the portfolio. The tech sector allocation came in
higher than target — 32% vs the 25% ceiling. Need to rebalance before
the end of the quarter to stay within the risk framework I set.

Risk tolerance this year is moderate. I'd rather underperform in a rally
than carry concentration risk into a downturn. The plan is to trim the
winners and add to the underweight positions in healthcare and energy.

Long-term horizon is 10-15 years to financial independence. The savings
rate target is 35% of gross income. Currently tracking at 31% — close
but need to tighten discretionary spending in the second half.
""",
        encoding="utf-8",
    )
    return p
