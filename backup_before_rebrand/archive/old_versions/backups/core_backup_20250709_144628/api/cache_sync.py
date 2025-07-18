"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Sync Service
==================

Periodically refreshes all registered API handlers and stores their
normalised JSON payloads into the local `flask/feeds/` cache hierarchy.

This service is **independent** from the trading loop - it can be run as
its own asyncio Task or integrated as a background task by the
`ApiIntegrationManager`.
"""

import asyncio
import importlib
import inspect
import logging
from pathlib import Path
from typing import List, ModuleType

from .handlers.base_handler import BaseAPIHandler

logger = logging.getLogger(__name__)

HANDLER_PACKAGE = "core.api.handlers"
DEFAULT_REFRESH: int = 300  # seconds


    class CacheSyncService:
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """Background service that refreshes all API handler caches."""

        def __init__(self, refresh_interval: int = DEFAULT_REFRESH) -> None:
        """Initialize the CacheSyncService.

            Args:
            refresh_interval: The interval in seconds to refresh the cache.
            """
            self.refresh_interval = refresh_interval
            self.handlers: List[BaseAPIHandler] = []
            self._task: asyncio.Task | None = None

                async def start(self) -> None:
                """Start the cache sync service."""
                    if self._task and not self._task.done():
                    logger.warning("CacheSyncService already running")
                return

                await self._discover_handlers()
                self._task = asyncio.create_task(self._run_loop())
                logger.info("[Service Started] CacheSyncService started with {0} handlers".format(len(self.handlers)))

                    async def stop(self) -> None:
                    """Stop the cache sync service."""
                        if self._task:
                        self._task.cancel()
                            try:
                            await self._task
                                except asyncio.CancelledError:
                            pass
                            self._task = None

                            # Close handler sessions
                            await asyncio.gather(*(h.close() for h in self.handlers), return_exceptions=True)
                            logger.info("[Service Stopped] CacheSyncService stopped")

                                async def _run_loop(self) -> None:
                                """The main loop that periodically refreshes the cache."""
                                    while True:
                                        try:
                                        await asyncio.gather(*(h.get_data(force_refresh=True) for h in self.handlers))
                                        except Exception as exc:  # noqa: BLE001
                                        logger.error("CacheSyncService iteration failed: %s", exc, exc_info=True)

                                        await asyncio.sleep(self.refresh_interval)

                                            async def _discover_handlers(self) -> None:
                                            """Dynamically import every module in `core.api.handlers` and register subclasses."""
                                            pkg = importlib.import_module(HANDLER_PACKAGE)
                                            pkg_path = Path(pkg.__file__).parent  # type: ignore[arg-type]

                                                for py_file in pkg_path.rglob("*.py"):
                                                    if py_file.name == "__init__.py" or py_file.name.startswith("_"):
                                                continue

                                                rel_mod = "{0}.{1}".format(HANDLER_PACKAGE, py_file.stem)
                                                    try:
                                                    mod: ModuleType = importlib.import_module(rel_mod)  # noqa: PERF401
                                                    except Exception as exc:  # noqa: BLE001
                                                    logger.error("Failed to import handler module %s: %s", rel_mod, exc)
                                                continue

                                                    for _, obj in inspect.getmembers(mod, inspect.isclass):
                                                        if issubclass(obj, BaseAPIHandler) and obj is not BaseAPIHandler:
                                                            try:
                                                            # type: ignore[call-arg]
                                                            handler: BaseAPIHandler = obj()
                                                            self.handlers.append(handler)
                                                            logger.info("Registered handler: %s", handler.NAME)
                                                            except Exception as exc:  # noqa: BLE001
                                                            logger.error("Failed to initialise handler %s: %s", obj, exc)
