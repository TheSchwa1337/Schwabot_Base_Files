from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import aiohttp
except ImportError:  # Fallback to requests for sync usage / testing
    aiohttp = None  # type: ignore

__all__ = ["BaseAPIHandler"]

logger = logging.getLogger(__name__)

"""Base API Handler

Provides an abstract base class for integrating third-party APIs into the
Schwabot data pipeline. Child classes only need to implement
`_fetch_raw()` and (optionally) `_parse_raw()` and the handler is ready
for use by the cache sync subsystem.
"""


class BaseAPIHandler(ABC):
    """Abstract base-class for external API handlers with enhanced features."""

    # --- Class-level configuration -------------------------------------------------

    NAME: str = "generic_api"  # Override in subclass (e.g. lunarcrush)

    CACHE_SUBDIR: str = "generic"  # flask/feeds/<CACHE_SUBDIR>/latest.json

    REFRESH_INTERVAL: int = 300  # seconds - 5-minute default

    # Rate limiting configuration
    RATE_LIMIT_REQUESTS: int = 100  # requests per window
    RATE_LIMIT_WINDOW: int = 3600  # window in seconds (1 hour)
    RATE_LIMIT_DELAY: float = 1.0  # delay between requests in seconds

    # Error handling configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    TIMEOUT: int = 30  # seconds

    # ------------------------------------------------------------------------------

    def __init__(self, cache_root: Path | str = Path("flask/feeds")) -> None:
        self.cache_root: Path = Path(cache_root)
        self._last_refresh: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None if aiohttp else None
        
        # Rate limiting state
        self._request_count: int = 0
        self._rate_limit_window_start: float = time.time()
        self._last_request_time: float = 0.0
        
        # Error tracking
        self._error_count: int = 0
        self._last_error_time: float = 0.0
        self._consecutive_errors: int = 0

    # Public API --------------------------------------------------------------------

    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return cached data, refreshing from the remote API if needed."""
        if force_refresh or (time.time() - self._last_refresh > self.REFRESH_INTERVAL):
            try:
                # Check rate limits before making request
                await self._check_rate_limit()
                
                raw = await self._fetch_raw()
                parsed = await self._parse_raw(raw)
                await self._write_cache(parsed)
                self._last_refresh = time.time()
                
                # Reset error counters on success
                self._consecutive_errors = 0
                self._error_count = 0
                
                return parsed
            except Exception as exc:
                self._handle_error(exc)
                logger.error("%s: refresh failed - %s", self.NAME, exc, exc_info=True)
                # Fallback to cached data if available
                return await self._read_cache()

        return await self._read_cache()  # Return cached data if no refresh needed

    async def fetch_data(self) -> Dict[str, Any]:
        """Alias for get_data() to ensure interface compliance."""
        return await self.get_data()

    def is_fresh(self) -> bool:
        """Check if cached data is fresh (within refresh interval)."""
        return (time.time() - self._last_refresh) <= self.REFRESH_INTERVAL

    def cache_hash(self) -> str:
        """Generate hash of cached data for integrity checking."""
        try:
            cached_data = asyncio.run(self._read_cache())
            if cached_data:
                return str(hash(json.dumps(cached_data, sort_keys=True)))
            return ""
        except Exception:
            return ""

    def entropy_value(self) -> float:
        """Calculate entropy value from cached data for strategy integration."""
        try:
            cached_data = asyncio.run(self._read_cache())
            if not cached_data:
                return 0.5  # Default entropy
            
            # Calculate entropy based on data variability
            if isinstance(cached_data, dict):
                # Use timestamp variance as entropy measure
                timestamps = []
                for value in cached_data.values():
                    if isinstance(value, (int, float)):
                        timestamps.append(float(value))
                
                if timestamps:
                    variance = sum((x - sum(timestamps)/len(timestamps))**2 for x in timestamps) / len(timestamps)
                    return min(1.0, max(0.0, variance / 1000.0))  # Normalize to [0, 1]
            
            return 0.5
        except Exception:
            return 0.5

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        window_elapsed = current_time - self._rate_limit_window_start
        
        if window_elapsed > self.RATE_LIMIT_WINDOW:
            # Reset window
            self._rate_limit_window_start = current_time
            self._request_count = 0
        
        return {
            "requests_used": self._request_count,
            "requests_remaining": max(0, self.RATE_LIMIT_REQUESTS - self._request_count),
            "window_remaining": max(0, self.RATE_LIMIT_WINDOW - window_elapsed),
            "rate_limit_exceeded": self._request_count >= self.RATE_LIMIT_REQUESTS
        }

    def get_error_status(self) -> Dict[str, Any]:
        """Get current error status."""
        return {
            "total_errors": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "last_error_time": self._last_error_time,
            "time_since_last_error": time.time() - self._last_error_time
        }

    # Abstract methods --------------------------------------------------------------

    @abstractmethod
    async def _fetch_raw(self) -> Any:  # pragma: no cover  implemented by subclass
        """Fetch raw data from the remote API (network call)."""
        pass  # Must be implemented by subclass

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        """Transform raw payload into a normalised JSON-serialisable dict.

        Sub-classes may override for custom parsing. The default
        implementation assumes the payload is already JSON-compatible.
        """
        return raw  # type: ignore[return-value]

    # Rate limiting helpers ---------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Check if we need to reset the window
        if current_time - self._rate_limit_window_start > self.RATE_LIMIT_WINDOW:
            self._rate_limit_window_start = current_time
            self._request_count = 0
        
        # Check if we've exceeded the rate limit
        if self._request_count >= self.RATE_LIMIT_REQUESTS:
            window_remaining = self.RATE_LIMIT_WINDOW - (current_time - self._rate_limit_window_start)
            if window_remaining > 0:
                logger.warning(f"{self.NAME}: Rate limit exceeded, waiting {window_remaining:.1f} seconds")
                await asyncio.sleep(window_remaining)
                self._rate_limit_window_start = current_time
                self._request_count = 0
        
        # Enforce delay between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.RATE_LIMIT_DELAY:
            delay_needed = self.RATE_LIMIT_DELAY - time_since_last
            await asyncio.sleep(delay_needed)
        
        self._request_count += 1
        self._last_request_time = time.time()

    def _handle_error(self, exc: Exception) -> None:
        """Handle and track errors."""
        self._error_count += 1
        self._consecutive_errors += 1
        self._last_error_time = time.time()
        
        # Log error with context
        logger.error(f"{self.NAME}: Error occurred (total: {self._error_count}, consecutive: {self._consecutive_errors}) - {exc}")

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
        logger.debug("%s: cache updated %s", self.NAME, path)

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
            timeout = aiohttp.ClientTimeout(total=self.TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)

        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # Validation methods ------------------------------------------------------------

    def validate_implementation(self) -> bool:
        """Validate that the handler implements all required methods."""
        required_methods = ['_fetch_raw', 'get_data', 'fetch_data']
        
        for method_name in required_methods:
            if not hasattr(self, method_name):
                logger.error(f"{self.NAME}: Missing required method '{method_name}'")
                return False
        
        return True

    def get_handler_info(self) -> Dict[str, Any]:
        """Get comprehensive handler information."""
        return {
            "name": self.NAME,
            "cache_subdir": self.CACHE_SUBDIR,
            "refresh_interval": self.REFRESH_INTERVAL,
            "rate_limit_requests": self.RATE_LIMIT_REQUESTS,
            "rate_limit_window": self.RATE_LIMIT_WINDOW,
            "max_retries": self.MAX_RETRIES,
            "timeout": self.TIMEOUT,
            "cache_file": str(self._cache_file),
            "is_fresh": self.is_fresh(),
            "rate_limit_status": self.get_rate_limit_status(),
            "error_status": self.get_error_status(),
            "implementation_valid": self.validate_implementation()
        }
