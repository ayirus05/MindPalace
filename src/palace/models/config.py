"""Configuration models for MindPalace Tier 2.

The :class:`PalaceConfig` hierarchy is the single source of truth for all
runtime settings.  It is loaded from ``config.yaml`` but can also be
constructed programmatically (useful in tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    timeout_seconds: float = 60.0
    batch_size: int = 32
    max_retries: int = 4
    retry_initial_wait_seconds: float = 1.0
    retry_max_wait_seconds: float = 30.0


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: int = 300
    overlap_tokens: int = 40
    tokens_per_word: float = 1.3


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "index/vector.lancedb"
    table_name: str = "chunks"
    metric: str = "cosine"


class IndexerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dirs: list[str] = Field(default_factory=lambda: ["journals"])
    extensions: list[str] = Field(default_factory=lambda: [".md", ".txt"])
    hash_cache_path: str = "cache/hash_cache.json"


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefilter_top_k: int = 50
    default_top_k: int = 8
    minimum_score: float = 0.15
    recency_boost_days: int = 90
    recency_boost_amount: float = 0.15
    deduplicate_by_source: bool = True
    domain_weights: dict[str, float] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    rich_output: bool = True
    file_path: str | None = "cache/palace.log"


class PalaceConfig(BaseModel):
    """Top-level configuration.  ``project_root`` is resolved at load time."""

    model_config = ConfigDict(extra="forbid")

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    project_root: Path = Path(".")

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to ``project_root`` (unless absolute)."""
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    def resolve_all(self, paths: list[str]) -> list[Path]:
        return [self.resolve(p) for p in paths]

    @model_validator(mode="after")
    def _validate_paths(self) -> PalaceConfig:
        # Coerce project_root to absolute + resolved.
        if not self.project_root.is_absolute():
            object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> PalaceConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to ``config.yaml``.
            project_root: Optional root for resolving relative paths.
                Defaults to the parent directory of the config file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        root = Path(project_root) if project_root else p.parent.resolve()
        return cls.model_validate({**raw, "project_root": root})

    @classmethod
    def default_for(cls, project_root: str | Path) -> PalaceConfig:
        """Create a default config without reading a file (used by tests)."""
        return cls.model_validate({"project_root": str(project_root)})
