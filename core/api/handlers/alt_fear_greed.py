from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional
from .base_handler import BaseAPIHandler

import aiohttp
import requests

try:
    pass
    except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    pass
    except ImportError:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

"""Alternative.me Fear & Greed Handler"

Fetches the latest Fear & Greed Index from https://alternative.me.

The API returns JSON with a list; we normalise it into a dict and cache
it under `flask/feeds/sentiment/fear_greed.json`.
"""

URL = "https://api.alternative.me/fng/?limit=1&format=json"


class FearGreedHandler(BaseAPIHandler):
    NAME = "fear_greed_index"
    CACHE_SUBDIR = "sentiment"
    REFRESH_INTERVAL = 600  # 10-minute updates are sufficient
    CACHE_EXPIRY = 3600  # 1-hour cache expiry
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, cache_root: str = "flask/feeds") -> None:
        """Initialize handler with enhanced cache management."""
        super().__init__(cache_root)
        self._last_cache_check = 0.0
        self._cache_expiry_time = 0.0

    async def _fetch_raw(self) -> Any:  # noqa: D401
        """Fetch raw JSON from the Alternative.me API with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                if aiohttp:
                    session = await self._get_session()
                    async with session.get(URL, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                elif requests:
                    # Blocking only used if aiohttp missing (e.g. quick, tests)
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, lambda: requests.get(URL, timeout=15).json())
                else:
                    raise RuntimeError("Neither aiohttp nor requests is available for HTTP calls")

            except Exception as exc:
                logger.warning("Attempt {0}/{1} failed: {2}".format(attempt + 1, self.MAX_RETRIES, exc))
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error()
                        "All {0} attempts failed for {1}".format()
                            self.MAX_RETRIES, 
                            self.NAME)
                    )
                    raise

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        """Normalise API payload into a simple dict with enhanced error handling."""
        try:
            if not isinstance(raw, dict) or "data" not in raw:
                raise ValueError("Invalid API response structure")

            data = raw["data"][0]

            # Validate required fields
            required_fields = ["value", "value_classification", "timestamp"]
            for field in required_fields:
                if field not in data:
                    raise ValueError("Missing required field: {0}".format(field))

            parsed_data = {}
                "value": int(data["value"]),
                "value_classification": data.get("value_classification", "Unknown"),
                "timestamp": int(data["timestamp"]),
                "time_until_update": int(raw.get("metadata", {}).get("time_until_update", 0)),
                "normalized_value": self._normalize_fear_greed_value(int(data["value"])),
                "sentiment_score": self._calculate_sentiment_score(int(data["value"])),
                "cache_timestamp": int(time.time()),
            }

            # Set cache expiry
            self._cache_expiry_time = time.time() + self.CACHE_EXPIRY

            return parsed_data

        except Exception as exc:
            logger.error("{0}: failed to parse payload - {1}".format(self.NAME, exc))
            # Return a fallback response structure
            return {}
                "value": 50,  # Neutral fear/greed value
                "value_classification": "Unknown",
                "timestamp": int(time.time()),
                "time_until_update": 0,
                "normalized_value": 0.5,
                "sentiment_score": 0.0,
                "cache_timestamp": int(time.time()),
                "error": str(exc),
            }

    def _normalize_fear_greed_value(self, value: int) -> float:
        """Normalize fear/greed value to [0, 1] range."""
        return max(0.0, min(1.0, value / 100.0))

    def _calculate_sentiment_score(self, value: int) -> float:
        """Calculate sentiment score based on fear/greed value."""
        # Fear < 30 = potential long entry (positive, sentiment)
        # Greed > 70 = potential short entry (negative, sentiment)
        if value < 30:
            return 1.0 - (value / 30.0)  # Higher score for lower fear
        elif value > 70:
            return -((value - 70) / 30.0)  # Negative score for higher greed
        else:
            return 0.0  # Neutral sentiment

    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return cached data, refreshing from the remote API if needed."""
        current_time = time.time()

        # Check cache expiry
        if current_time > self._cache_expiry_time:
            force_refresh = True
            logger.debug("{0}: Cache expired, forcing refresh".format(self.NAME))

        if force_refresh or (current_time - self._last_refresh > self.REFRESH_INTERVAL):
            try:
                raw = await self._fetch_raw()
                parsed = await self._parse_raw(raw)
                await self._write_cache(parsed)
                self._last_refresh = current_time
                return parsed
            except Exception as exc:
                logger.error("{0}: refresh failed - {1}".format(self.NAME, exc), exc_info=True)
                # Fallback to cached data if available
                cached_data = await self._read_cache()
                if cached_data:
                    logger.info("{0}: Using cached data due to refresh failure".format(self.NAME))
                    return cached_data
                else:
                    # Return default data if no cache available
                    return {}
                        "value": 50,
                        "value_classification": "Neutral",
                        "timestamp": int(current_time),
                        "time_until_update": 0,
                        "normalized_value": 0.5,
                        "sentiment_score": 0.0,
                        "cache_timestamp": int(current_time),
                        "error": "No data available",
                    }

        return await self._read_cache()  # Return cached data if no refresh needed

    def is_cache_fresh(self) -> bool:
        """Check if cached data is still fresh."""
        return time.time() <= self._cache_expiry_time

    def get_cache_age(self) -> float:
        """Get age of cached data in seconds."""
        return time.time() - self._last_refresh
