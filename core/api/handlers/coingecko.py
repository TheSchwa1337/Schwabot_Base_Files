#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoinGecko API Handler
====================

Fetches market data, prices, and broader market sentiment from CoinGecko API.
Provides comprehensive market data for multiple cryptocurrencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import asyncio
import time

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .base_handler import BaseAPIHandler

logger = logging.getLogger(__name__)

# CoinGecko API configuration
BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoHandler(BaseAPIHandler):
    NAME = "coingecko"
    CACHE_SUBDIR = "market_data"
    REFRESH_INTERVAL = 300  # 5-minute updates for market data

    def __init__(self, api_key: str = None, cache_root: str = "flask/feeds"):
        super().__init__(cache_root)
        self.api_key = api_key  # CoinGecko has free tier without API key

        # Coins to track
        self.coins = [
            "bitcoin",
            "ethereum",
            "binancecoin",
            "cardano",
            "solana",
            "ripple",
            "polkadot",
            "dogecoin",
            "avalanche-2",
            "chainlink",
        ]

        # Currencies for price conversion
        self.vs_currencies = ["usd", "btc", "eth"]

    async def _fetch_raw(self) -> Any:
        """Fetch raw market data from CoinGecko API."""
        all_data = {}

        # Headers with API key if available
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            # Fetch global market data
            global_data = await self._fetch_global_data(headers)
            all_data["global"] = global_data

            # Fetch price data for tracked coins
            price_data = await self._fetch_price_data(headers)
            all_data["prices"] = price_data

            # Fetch trending coins
            trending_data = await self._fetch_trending_data(headers)
            all_data["trending"] = trending_data

            # Fetch market dominance
            dominance_data = await self._fetch_dominance_data(headers)
            all_data["dominance"] = dominance_data

        except Exception as e:
            logger.error(f"Failed to fetch CoinGecko data: {e}")
            all_data = {"global": {}, "prices": {}, "trending": {}, "dominance": {}}

        return all_data

    async def _fetch_global_data(self, headers: Dict) -> Dict:
        """Fetch global cryptocurrency market data."""
        try:
            if aiohttp:
                session = await self._get_session()
                async with session.get(f"{BASE_URL}/global", headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            elif requests:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{BASE_URL}/global", headers=headers, timeout=15
                    ),
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch global data: {e}")
            return {}

    async def _fetch_price_data(self, headers: Dict) -> Dict:
        """Fetch price data for tracked coins."""
        try:
            coins_str = ",".join(self.coins)
            vs_currencies_str = ",".join(self.vs_currencies)

            params = {
                "ids": coins_str,
                "vs_currencies": vs_currencies_str,
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            }

            if aiohttp:
                session = await self._get_session()
                async with session.get(
                    f"{BASE_URL}/simple/price", params=params, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            elif requests:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{BASE_URL}/simple/price",
                        params=params,
                        headers=headers,
                        timeout=15,
                    ),
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return {}

    async def _fetch_trending_data(self, headers: Dict) -> Dict:
        """Fetch trending coins data."""
        try:
            if aiohttp:
                session = await self._get_session()
                async with session.get(
                    f"{BASE_URL}/search/trending", headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            elif requests:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{BASE_URL}/search/trending", headers=headers, timeout=15
                    ),
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch trending data: {e}")
            return {}

    async def _fetch_dominance_data(self, headers: Dict) -> Dict:
        """Calculate market dominance data."""
        try:
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 10,
                "page": 1,
                "sparkline": "false",
            }

            if aiohttp:
                session = await self._get_session()
                async with session.get(
                    f"{BASE_URL}/coins/markets", params=params, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            elif requests:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{BASE_URL}/coins/markets",
                        params=params,
                        headers=headers,
                        timeout=15,
                    ),
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch dominance data: {e}")
            return []

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        """Parse CoinGecko data into normalized format."""
        try:
            parsed_data = {
                "timestamp": int(time.time()),
                "global_metrics": {},
                "coin_prices": {},
                "trending_coins": [],
                "market_dominance": {},
                "market_sentiment": {},
            }

            # Parse global data
            if "global" in raw and "data" in raw["global"]:
                global_data = raw["global"]["data"]
                parsed_data["global_metrics"] = {
                    "total_market_cap_usd": global_data.get("total_market_cap", {}).get(
                        "usd", 0
                    ),
                    "total_volume_24h_usd": global_data.get("total_volume", {}).get(
                        "usd", 0
                    ),
                    "market_cap_change_24h": global_data.get(
                        "market_cap_change_percentage_24h_usd", 0
                    ),
                    "active_cryptocurrencies": global_data.get(
                        "active_cryptocurrencies", 0
                    ),
                    "markets": global_data.get("markets", 0),
                    "market_cap_percentage": global_data.get(
                        "market_cap_percentage", {}
                    ),
                }

            # Parse price data
            if "prices" in raw:
                for coin_id, price_info in raw["prices"].items():
                    if isinstance(price_info, dict):
                        parsed_data["coin_prices"][coin_id] = {
                            "usd_price": price_info.get("usd", 0),
                            "btc_price": price_info.get("btc", 0),
                            "eth_price": price_info.get("eth", 0),
                            "usd_market_cap": price_info.get("usd_market_cap", 0),
                            "usd_24h_vol": price_info.get("usd_24h_vol", 0),
                            "usd_24h_change": price_info.get("usd_24h_change", 0),
                            "last_updated": price_info.get("last_updated_at", 0),
                        }

            # Parse trending data
            if "trending" in raw and "coins" in raw["trending"]:
                for coin_data in raw["trending"]["coins"]:
                    if "item" in coin_data:
                        coin = coin_data["item"]
                        parsed_data["trending_coins"].append(
                            {
                                "id": coin.get("id", ""),
                                "name": coin.get("name", ""),
                                "symbol": coin.get("symbol", ""),
                                "market_cap_rank": coin.get("market_cap_rank", 0),
                                "price_btc": coin.get("price_btc", 0),
                            }
                        )

            # Parse dominance data
            if "dominance" in raw and isinstance(raw["dominance"], list):
                total_market_cap = sum(
                    coin.get("market_cap", 0) for coin in raw["dominance"][:10]
                )
                for coin in raw["dominance"][:10]:
                    coin_id = coin.get("id", "")
                    market_cap = coin.get("market_cap", 0)
                    dominance_percent = (
                        (market_cap / total_market_cap * 100)
                        if total_market_cap > 0
                        else 0
                    )
                    parsed_data["market_dominance"][coin_id] = {
                        "market_cap": market_cap,
                        "dominance_percentage": dominance_percent,
                        "rank": coin.get("market_cap_rank", 0),
                    }

            # Calculate market sentiment indicators
            parsed_data["market_sentiment"] = self._calculate_market_sentiment(
                parsed_data
            )

            return parsed_data

        except Exception as exc:
            logger.error("%s: failed to parse CoinGecko data – %s", self.NAME, exc)
            return {
                "timestamp": int(time.time()),
                "global_metrics": {},
                "coin_prices": {},
                "trending_coins": [],
                "market_dominance": {},
                "market_sentiment": {},
            }

    def _calculate_market_sentiment(self, data: Dict) -> Dict[str, float]:
        """Calculate market sentiment indicators."""
        sentiment = {}

        try:
            # Overall market sentiment based on 24h changes
            positive_changes = 0
            negative_changes = 0
            total_volume = 0

            for coin_id, price_data in data.get("coin_prices", {}).items():
                change_24h = price_data.get("usd_24h_change", 0)
                volume_24h = price_data.get("usd_24h_vol", 0)

                if change_24h > 0:
                    positive_changes += 1
                elif change_24h < 0:
                    negative_changes += 1

                total_volume += volume_24h

            total_coins = positive_changes + negative_changes
            if total_coins > 0:
                bullish_ratio = positive_changes / total_coins
            else:
                bullish_ratio = 0.5

            sentiment["bullish_ratio"] = bullish_ratio
            sentiment["total_volume_24h"] = total_volume

            # Bitcoin dominance sentiment
            btc_dominance = (
                data.get("global_metrics", {})
                .get("market_cap_percentage", {})
                .get("btc", 50)
            )
            if btc_dominance > 60:
                sentiment["btc_dominance_signal"] = "strong"
            elif btc_dominance > 45:
                sentiment["btc_dominance_signal"] = "neutral"
            else:
                sentiment["btc_dominance_signal"] = "weak"

            # Market cap change sentiment
            market_cap_change = data.get("global_metrics", {}).get(
                "market_cap_change_24h", 0
            )
            if market_cap_change > 5:
                sentiment["market_trend"] = "very_bullish"
            elif market_cap_change > 2:
                sentiment["market_trend"] = "bullish"
            elif market_cap_change > -2:
                sentiment["market_trend"] = "neutral"
            elif market_cap_change > -5:
                sentiment["market_trend"] = "bearish"
            else:
                sentiment["market_trend"] = "very_bearish"

        except Exception as e:
            logger.error(f"Failed to calculate market sentiment: {e}")
            sentiment = {
                "bullish_ratio": 0.5,
                "total_volume_24h": 0,
                "btc_dominance_signal": "neutral",
                "market_trend": "neutral",
            }

        return sentiment
