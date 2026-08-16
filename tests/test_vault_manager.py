from __future__ import annotations

from palace.models.config import PalaceConfig
from palace.vault.manager import VaultManager


def test_vault_manager_is_concrete_and_persists_fields(tmp_path) -> None:
    config = PalaceConfig.default_for(tmp_path)
    manager = VaultManager(config)

    manager.write_field("alpha", "count", 3)

    assert manager.read_field("alpha", "count") == 3
    assert manager.list_lockers("all") == ["alpha"]
