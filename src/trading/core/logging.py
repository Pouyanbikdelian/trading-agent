"""Logging setup. One configure call per process; subsequent calls are no-ops.

Logs go to:
* stderr (pretty) with INFO+ by default,
* a rotating file under ``settings.log_dir`` with DEBUG+ for forensics.

We use loguru rather than stdlib logging — it's a single sink with sane
defaults, structured context via ``.bind()``, and no handler boilerplate.
"""

from __future__ import annotations

import sys

from loguru import logger

from trading.core.config import settings

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure global logging. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.ensure_dirs()
    logger.remove()  # drop loguru's default handler

    # Pretty console handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,  # don't leak local variables to logs
        enqueue=False,
    )

    # Rotating file handler for full DEBUG forensics
    logger.add(
        settings.log_dir / "trading.{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=False,
        enqueue=True,  # process-safe; survives crashes mid-write
    )

    _CONFIGURED = True


__all__ = ["configure_logging", "logger"]
