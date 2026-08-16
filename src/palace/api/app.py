"""FastAPI application for inspecting and updating Core Vault lockers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from palace.models.config import PalaceConfig
from palace.vault.manager import VaultManager


LockerType = Literal["all", "system", "user"]


class LockerFieldUpdate(BaseModel):
    """Request body for creating or replacing one locker field."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    value: Any


def create_app(config: PalaceConfig) -> FastAPI:
    """Create a configured Vault API application.

    The manager is constructed once and retained on application state, keeping
    endpoint functions independent of filesystem construction details.
    """
    app = FastAPI(title="MindPalace Core Vault API", version="1.0.0")
    app.state.config = config
    app.state.vault_manager = VaultManager(config)

    @app.get("/api/lockers", response_model=list[str])
    async def list_lockers(
        locker_type: LockerType = Query(default="all"),
        manager: VaultManager = Depends(_get_vault_manager),
    ) -> list[str]:
        """List lockers in one or both configured scopes."""
        return manager.list_lockers(locker_type)

    @app.get("/api/lockers/{locker_name}", response_model=dict[str, Any])
    async def get_locker(
        locker_name: str,
        request: Request,
        manager: VaultManager = Depends(_get_vault_manager),
    ) -> dict[str, Any]:
        """Return the complete JSON object stored for a locker."""
        return _read_locker(request.app.state.config, manager, locker_name)

    @app.post("/api/lockers/{locker_name}", response_model=dict[str, Any])
    async def update_locker(
        locker_name: str,
        update: LockerFieldUpdate,
        request: Request,
        manager: VaultManager = Depends(_get_vault_manager),
    ) -> dict[str, Any]:
        """Create or replace one locker field and return the updated locker."""
        try:
            manager.write_field(locker_name, update.field, update.value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return _read_locker(request.app.state.config, manager, locker_name)

    return app


async def _get_vault_manager(request: Request) -> VaultManager:
    """Resolve the app-scoped manager dependency."""
    return cast(VaultManager, request.app.state.vault_manager)


def _read_locker(
    config: PalaceConfig,
    manager: VaultManager,
    locker_name: str,
) -> dict[str, Any]:
    """Read a full locker object using the manager's system-first ordering."""
    if locker_name not in manager.list_lockers("all"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Locker not found: {locker_name}",
        )

    paths = (
        config.resolve(config.vault.system_lockers_dir) / f"{locker_name}.json",
        config.resolve(config.vault.user_lockers_dir) / f"{locker_name}.json",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:  # The filesystem changed after list_lockers().
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Locker not found: {locker_name}",
        )

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read locker: {locker_name}",
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Locker is not a JSON object: {locker_name}",
        )
    return value


__all__ = ["LockerFieldUpdate", "LockerType", "create_app"]
