"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from palace.models.config import PalaceConfig


class TestPalaceConfig:
    def test_default_for(self, tmp_path: Path) -> None:
        cfg = PalaceConfig.default_for(tmp_path)
        assert cfg.project_root == tmp_path.resolve()
        assert cfg.embedding.model == "nomic-embed-text"
        assert cfg.chunker.target_tokens == 300
        assert cfg.database.table_name == "chunks"

    def test_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump({
                "embedding": {"model": "custom-model", "batch_size": 8},
                "search": {"default_top_k": 12, "minimum_score": 0.2},
            }),
            encoding="utf-8",
        )
        cfg = PalaceConfig.from_yaml(cfg_path)
        assert cfg.embedding.model == "custom-model"
        assert cfg.embedding.batch_size == 8
        assert cfg.search.default_top_k == 12
        assert cfg.search.minimum_score == 0.2
        # project_root defaults to the parent of the config file.
        assert cfg.project_root == tmp_path.resolve()

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PalaceConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_resolve_relative(self, tmp_path: Path) -> None:
        cfg = PalaceConfig.default_for(tmp_path)
        assert cfg.resolve("journals") == (tmp_path / "journals").resolve()

    def test_resolve_absolute(self, tmp_path: Path) -> None:
        cfg = PalaceConfig.default_for(tmp_path)
        abs_path = "/absolute/path"
        assert cfg.resolve(abs_path) == Path(abs_path)

    def test_resolve_all(self, tmp_path: Path) -> None:
        cfg = PalaceConfig.default_for(tmp_path)
        paths = cfg.resolve_all(["journals", "notes"])
        assert len(paths) == 2
        assert paths[0] == (tmp_path / "journals").resolve()

    def test_invalid_field_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            PalaceConfig.model_validate({"embedding": {"batch_size": -1}})

    def test_project_root_becomes_absolute(self, tmp_path: Path) -> None:
        rel = Path(tmp_path.name)
        cfg = PalaceConfig.model_validate({"project_root": str(rel)})
        assert cfg.project_root.is_absolute() or cfg.project_root == rel.resolve()
