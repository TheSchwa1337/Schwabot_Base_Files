#!/usr/bin/env python3
"""Centralised logging configuration for Schwabot.

This tiny utility gives all modules a consistent way to obtain a
`logging.Logger` with sensible defaults while remaining fully
Flake-8-compliant and Windows-CLI-friendly.

Usage
-----
>>> from utils.logging_setup import configure_logging, set_level
>>> logger = configure_logging(__name__, level="DEBUG")
>>> logger.info("Message")
"""
from __future__ import annotations

import logging
from typing import Any

__all__: list[str] = [
    "configure_logging",
    "set_level",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def configure_logging(
    name: str | None = None,
    *,
    level: str | int = "INFO",
    fmt: str | None = None,
    datefmt: str | None = None,
    **basic_cfg: Any,
) -> logging.Logger:
    """Return a module-level logger configured once per process.

    Parameters
    ----------
    name:
        Logger name – typically ``__name__``.
    level:
        Text or numeric log-level (e.g. ``"DEBUG"`` or ``logging.DEBUG``).
    fmt, datefmt:
        Custom ``logging.basicConfig`` format strings.  If omitted, we supply
        Schwabot’s default format.
    **basic_cfg:
        Any extra keyword arguments are forwarded to
        ``logging.basicConfig`` allowing colour handlers, file handlers, …
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = int(level)

    if not logging.getLogger().handlers:
        # Root not configured yet – set it up
        logging.basicConfig(
            level=numeric_level,
            format=fmt
            or "% (asctime)s | %(levelname)-8s | %(name)s: %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
            **basic_cfg,
        )
    else:
        # Root already configured – just adjust level if asked
        logging.getLogger().setLevel(numeric_level)

    return logging.getLogger(name or __name__)


def set_level(level: str | int) -> None:
    """Convenience wrapper to raise / lower global log-level at runtime."""
    root = logging.getLogger()
    if isinstance(level, str):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
    else:
        root.setLevel(int(level))
