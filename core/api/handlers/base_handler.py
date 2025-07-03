from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp  # Preferred for non-blocking IO

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Base API Handler ====================

Provides an abstract base class for integrating third-party APIs into the
Schwabot data pipeline.  Child classes only need to implement
`_fetch_raw()` and (optionally) `_parse_raw()` and the handler is ready
for use by the cache sync subsystem.
"""

try:
    import aiohttp  # type: ignore
except ImportError:  # Fallback to requests for sync usage / testing
    aiohttp = None  # type: ignore

__all__ = ["BaseAPIHandler"]

logger = logging.getLogger(__name__)


class BaseAPIHandler(ABC):
    """Abstract base-class for external API handlers."""

    # --- Class-level configuration -------------------------------------------------
    NAME: str = "generic_api"  # Override in subclass (e.g. lunarcrush)
    CACHE_SUBDIR: str = "generic"  # flask/feeds/<CACHE_SUBDIR>/latest.json
    REFRESH_INTERVAL: int = 300  # seconds - 5-minute default

    # ------------------------------------------------------------------------------

    def __init__(self, cache_root: Path | str = Path("flask/feeds")) -> None:
        self.cache_root: Path = Path(cache_root)
        self._last_refresh: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None if aiohttp else None

    # Public API --------------------------------------------------------------------
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return cached data, refreshing from the remote API if needed."""
        if force_refresh or (time.time() - self._last_refresh > self.REFRESH_INTERVAL):
            try:
                raw = await self._fetch_raw()
                parsed = await self._parse_raw(raw)
                await self._write_cache(parsed)
                self._last_refresh = time.time()
                return parsed
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: refresh failed - %s", self.NAME, exc, exc_info=True)
                # Fallback to cached data if available
                return await self._read_cache()
        return await self._read_cache()  # Return cached data if no refresh needed

    # Abstract methods --------------------------------------------------------------
    @abstractmethod
    async def _fetch_raw(self) -> Any:  # pragma: no cover – implemented by subclass
        """Fetch raw data from the remote API (network call)."""
        pass  # Must be implemented by subclass

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        """Transform raw payload into a normalised JSON-serialisable dict.

        Sub-classes may override for custom parsing.  The default
        implementation assumes the payload is already JSON-compatible.
        """
        return raw  # type: ignore[return-value]

    # Caching helpers ---------------------------------------------------------------
    @property
    def _cache_file(self) -> Path:
        return self.cache_root / self.CACHE_SUBDIR / "latest.json"

    async def _write_cache(self, data: Dict[str, Any]) -> None:
        path = self._cache_file
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use thread-pool to avoid blocking event-loop for file IO
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, path.write_text, json.dumps(data, indent=2))
        logger.debug("%s: cache updated → %s", self.NAME, path)

    async def _read_cache(self) -> Dict[str, Any]:
        path = self._cache_file
        if not path.exists():
            logger.warning("%s: cache not found (%s)", self.NAME, path)
            return {}
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, path.read_text)
        return json.loads(text)

    # Session helper for aiohttp ----------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:  # type: ignore[return-type]
        if not aiohttp:
            raise RuntimeError("aiohttp is required for async HTTP but not installed")
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
