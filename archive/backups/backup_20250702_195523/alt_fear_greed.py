from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

import aiohttp  # type: ignore
import requests  # type: ignore

from .base_handler import BaseAPIHandler

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Alternative.me Fear & Greed Handler ==================================

Fetches the latest Fear & Greed Index from https://alternative.me.
The API returns JSON with a list; we normalise it into a dict and cache
it under `flask/feeds/sentiment/fear_greed.json`.

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

URL = https://api.alternative.me/fng/?limit=1&format=json


class FearGreedHandler(BaseAPIHandler):
    NAME =  fear_greed_index
    CACHE_SUBDIR =  sentimentREFRESH_INTERVAL = 600  # 10-minute updates are sufficient

    async def _fetch_raw(self) -> Any:  # noqa: D401
        Fetch raw JSON from the Alternative.me API.if aiohttp: session = await self._get_session()
            async with session.get(URL) as resp:
                resp.raise_for_status()
                return await resp.json()
        elif requests:
            # Blocking – only used if aiohttp missing (e.g. quick tests)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: requests.get(URL, timeout=15).json())
        else:
            raise RuntimeError(Neither aiohttp nor requests is available for HTTP calls)

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        Normalise API payload into a simple dict.try: data = raw[data][0]
            return {value: int(data[value]),value_classification: data.get(value_classification,Unknown),timestamp": int(data[timestamp]),time_until_update": int(raw.get(metadata", {}).get(time_until_update, 0)),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(%s: failed to parse payload – %s, self.NAME, exc)
            # Return a fallback response structure
            return {value: 50,  # Neutral fear/greed value
                value_classification:Unknown,timestamp: int(time.time()),time_until_update": 0,error: str(exc),
            }
