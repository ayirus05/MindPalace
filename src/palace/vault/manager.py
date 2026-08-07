"""Persistence interface and JSON implementation for the Core Vault.

Each locker is a human-editable JSON object stored in either the system or
user locker directory.  Field names are matched exactly; this subsystem does
not perform embedding, indexing, or fuzzy lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from palace.models.config import PalaceConfig


@runtime_checkable
class VaultManager(Protocol):
    """Storage contract used by Core Vault clients."""

    def read_field(self, locker_name: str, field_key: str) -> Any:
        """Return a field value, or ``None`` when the locker or field is absent."""
        ...

    def write_field(self, locker_name: str, field_key: str, value: Any) -> None:
        """Create or update one field in a locker."""
        ...

    def list_lockers(self, locker_type: str = "all") -> list[str]:
        """List locker names in the ``system``, ``user``, or ``all`` scope."""
        ...


class FileVaultManager:
    """JSON-file implementation of :class:`VaultManager`.

    System lockers take precedence when the same name exists in both scopes.
    Writes update an existing system locker when present; all new lockers are
    created in the user directory.

    Args:
        config: Application configuration containing both locker paths.
        base_dir: Optional base directory for resolving the configured paths.
            By default, paths are resolved relative to ``config.project_root``.
    """

    def __init__(
        self,
        config: PalaceConfig,
        base_dir: str | Path | None = None,
    ) -> None:
        root = Path(base_dir).resolve() if base_dir is not None else config.project_root
        self._system_dir = self._resolve(root, config.vault.system_lockers_dir)
        self._user_dir = self._resolve(root, config.vault.user_lockers_dir)
        self._system_dir.mkdir(parents=True, exist_ok=True)
        self._user_dir.mkdir(parents=True, exist_ok=True)

    def read_field(self, locker_name: str, field_key: str) -> Any:
        """Read an exact field from the first matching locker."""
        filename = self._locker_filename(locker_name)
        for directory in (self._system_dir, self._user_dir):
            path = directory / filename
            if path.is_file():
                data = self._load_object(path)
                if field_key in data:
                    return data[field_key]
        return None

    def write_field(self, locker_name: str, field_key: str, value: Any) -> None:
        """Write an exact field, creating a user locker when necessary.

        ``value`` must be JSON serializable.  Existing unrelated fields are
        preserved.
        """
        filename = self._locker_filename(locker_name)
        system_path = self._system_dir / filename
        path = system_path if system_path.is_file() else self._user_dir / filename
        data = self._load_object(path) if path.is_file() else {}
        data[field_key] = value

        # Serialize before opening the file so an invalid value cannot damage
        # an existing, human-edited locker.
        serialized = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{serialized}\n", encoding="utf-8")

    def list_lockers(self, locker_type: str = "all") -> list[str]:
        """Return unique locker names, sorted alphabetically."""
        directories: tuple[Path, ...]
        if locker_type == "system":
            directories = (self._system_dir,)
        elif locker_type == "user":
            directories = (self._user_dir,)
        elif locker_type == "all":
            directories = (self._system_dir, self._user_dir)
        else:
            raise ValueError(
                "locker_type must be one of 'all', 'system', or 'user'; "
                f"got {locker_type!r}"
            )

        names = {
            path.stem
            for directory in directories
            for path in directory.glob("*.json")
            if path.is_file()
        }
        return sorted(names)

    @staticmethod
    def _resolve(root: Path, configured_path: str) -> Path:
        path = Path(configured_path)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    @staticmethod
    def _locker_filename(locker_name: str) -> str:
        """Validate a logical locker name and append the JSON suffix."""
        name = locker_name.strip()
        candidate = Path(name)
        if (
            not name
            or candidate.is_absolute()
            or len(candidate.parts) != 1
            or name in {".", ".."}
        ):
            raise ValueError(f"Invalid locker name: {locker_name!r}")
        if candidate.suffix not in {"", ".json"}:
            raise ValueError("locker_name must not contain a non-JSON extension")
        return candidate.with_suffix(".json").name

    @staticmethod
    def _load_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in vault locker {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Vault locker must contain a JSON object: {path}")
        return value


__all__ = ["VaultManager", "FileVaultManager"]
