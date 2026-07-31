"""Logging setup using Rich for readable console output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    rich_output: bool = True,
    file_path: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure a single library-wide logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        rich_output: If True, use Rich's console handler for pretty output.
            If False, fall back to a plain StreamHandler.
        file_path: Optional path to a log file.
        force: If True, re-configure even if we already have.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger("palace")

    logger = logging.getLogger("palace")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    if file_path is not None:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    console_handler: logging.Handler
    if rich_output:
        console_handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
    logger.addHandler(console_handler)

    _CONFIGURED = True
    return logger


def configure_from_config(config_path: Optional[Path] = None) -> logging.Logger:
    """Convenience helper used when config isn't yet loaded."""
    return configure_logging()
