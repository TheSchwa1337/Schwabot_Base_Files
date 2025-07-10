import asyncio
import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import List

from .handlers.base_handler import BaseAPIHandler

""A service for synchronizing API data caches.from __future__ import annotations




# # Cache Sync Service
# ==================
#
# Periodically refreshes all registered API handlers and stores their
# normalised JSON payloads into the local `flask/feeds/` cache hierarchy.
#
# This service is **independent** from the trading loop – it can be run as
# its own asyncio Task or integrated as a background task by the
# `ApiIntegrationManager`.
#

logger = logging.getLogger(__name__)

HANDLER_PACKAGE = core.api.handlers
DEFAULT_REFRESH: int = 300  # seconds


class CacheSyncService:
    Background service that refreshes all API handler caches.def __init__():-> None:
        Initialize the CacheSyncService.

        Args:
            refresh_interval: The interval in seconds to refresh the cache.self.refresh_interval = refresh_interval
        self.handlers: List[BaseAPIHandler] = []
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------------
    async def start():-> None:
        Start the cache sync service.if self._task and not self._task.done():
            logger.warning(CacheSyncService already running)
            return await self._discover_handlers()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(🚀 CacheSyncService started with %d handlers, len(self.handlers))

    async def stop():-> None:Stop the cache sync service.if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Close handler sessions
        await asyncio.gather(*(h.close() for h in self.handlers), return_exceptions=True)
        logger.info(🛑 CacheSyncService stopped)

    # ------------------------------------------------------------------
    async def _run_loop():-> None:The main loop that periodically refreshes the cache.while True:
            try:
                await asyncio.gather(*(h.get_data(force_refresh = True) for h in self.handlers))
            except Exception as exc:  # noqa: BLE001
                logger.error(CacheSyncService iteration failed: %s, exc, exc_info = True)
            await asyncio.sleep(self.refresh_interval)

    # ------------------------------------------------------------------
    async def _discover_handlers():-> None:
        Dynamically import every module in `core.api.handlers` and register subclasses.pkg = importlib.import_module(HANDLER_PACKAGE)
        pkg_path = Path(pkg.__file__).parent  # type: ignore[arg-type]
        for py_file in pkg_path.rglob(*.py):
            if py_file.name == __init__.py or py_file.name.startswith(_):
                continue
            rel_mod = f{HANDLER_PACKAGE}.{py_file.stem}
            try:
                mod: ModuleType = importlib.import_module(rel_mod)  # noqa: PERF401
            except Exception as exc:  # noqa: BLE001
logger.error(Failed to import handler module %s: %s
logger.error(Failed to import rel_mod
logger.error(Failed to import exc)
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseAPIHandler) and obj is not BaseAPIHandler:
                    try:
                        handler: BaseAPIHandler = obj()  # type: ignore[call-arg]
                        self.handlers.append(handler)
                        logger.info(Registered handler: %s, handler.NAME)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(Failed to initialise handler %s: %s, obj, exc)
))