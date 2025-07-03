"""CoinGecko API Handler

Handles data fetching and processing from CoinGecko API endpoints
for cryptocurrency market data, price feeds, and market analytics.

This module provides both sync and async interfaces for:
- Price data fetching
- Market cap information
- Historical data retrieval
- Volume and trading metrics







Fetches comprehensive cryptocurrency market data from CoinGecko API.



Provides price data, market metrics, trending coins, and market dominance data.



"""

import asyncio
from typing import Any, Dict


from .base_handler import BaseAPIHandler

try:



except ImportError:

    aiohttp = None


try:



except ImportError:

    requests = None


logger = logging.getLogger(__name__)


# CoinGecko API configuration


BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoHandler(BaseAPIHandler):
    """CoinGecko API handler for comprehensive cryptocurrency market data."""

    NAME = "coingecko"

    CACHE_SUBDIR = "market_data"

    REFRESH_INTERVAL = 300  # 5-minute updates for market data

    def __init__(self, api_key: str = None, cache_root: str = "flask/feeds") -> None:
        """Initialize CoinGecko handler with API key and cache configuration."""

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
                    lambda: requests.get(f"{BASE_URL}/global", headers=headers, timeout=15),
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
                "include_market_cap": True,
                "include_24hr_vol": True,
                "include_24hr_change": True,
                "include_last_updated_at": True,
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

                async with session.get(f"{BASE_URL}/search/trending", headers=headers) as resp:

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
                "sparkline": False,
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
                    "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd", 0),
                    "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd", 0),
                    "market_cap_change_24h": global_data.get(
                        "market_cap_change_percentage_24h_usd", 0
                    ),
                    "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0),
                    "markets": global_data.get("markets", 0),
                    "defi_volume_24h": global_data.get("defi_volume_24h", 0),
                    "defi_market_cap": global_data.get("defi_market_cap", 0),
                }

            # Parse price data

            if "prices" in raw:

                parsed_data["coin_prices"] = raw["prices"]

            # Parse trending data

            if "trending" in raw and "coins" in raw["trending"]:

                trending_coins = []

                for coin in raw["trending"]["coins"]:

                    trending_coins.append(
                        {
                            "id": coin["item"]["id"],
                            "name": coin["item"]["name"],
                            "symbol": coin["item"]["symbol"],
                            "market_cap_rank": coin["item"]["market_cap_rank"],
                            "score": coin["item"]["score"],
                        }
                    )

                parsed_data["trending_coins"] = trending_coins

            # Parse dominance data

            if "dominance" in raw and isinstance(raw["dominance"], list):

                dominance_data = {}

                total_market_cap = sum(coin.get("market_cap", 0) for coin in raw["dominance"])

                for coin in raw["dominance"]:

                    if total_market_cap > 0:

                        dominance_pct = (coin.get("market_cap", 0) / total_market_cap) * 100

                        dominance_data[coin["symbol"]] = dominance_pct

                parsed_data["market_dominance"] = dominance_data

            # Calculate market sentiment

            parsed_data["market_sentiment"] = self._calculate_market_sentiment(parsed_data)

            return parsed_data

        except Exception as exc:  # noqa: BLE001

            logger.error("%s: failed to parse payload  %s", self.NAME, exc)

            # Return a fallback response structure

            return {
                "timestamp": int(time.time()),
                "global_metrics": {},
                "coin_prices": {},
                "trending_coins": [],
                "market_dominance": {},
                "market_sentiment": {},
                "error": str(exc),
            }

    def _calculate_market_sentiment(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates a simple market sentiment based on trending coins and market cap change."""

        sentiment_score = 0

        # Example: Simple calculation based on trending coins and global market cap change

        if "trending_coins" in parsed_data:

            sentiment_score += len(parsed_data["trending_coins"]) * 5

        if (
            "global_metrics" in parsed_data
            and "market_cap_change_24h" in parsed_data["global_metrics"]
        ):

            change = parsed_data["global_metrics"]["market_cap_change_24h"]

            if change > 0:

                sentiment_score += min(int(change / 10), 20)

            else:

                sentiment_score -= min(int(abs(change) / 10), 20)

        # Normalize to a range, e.g., -100 to 100 or 0 to 100

        # For simplicity, let's just cap it for now

        sentiment_score = max(-100, min(100, sentiment_score))

        return {
            "score": sentiment_score,
            "interpretation": (
                "Positive"
                if sentiment_score > 0
                else "Negative" if sentiment_score < 0 else "Neutral"
            ),
        }


# Example usage (for testing purposes)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    handler = CoinGeckoHandler()

    async def main():

        data = await handler.get_data(force_refresh=True)

        if data:

            logger.info("Successfully fetched and parsed CoinGecko data.")

            logger.info(json.dumps(data, indent=2))

        else:

            logger.error("Failed to fetch CoinGecko data.")

    asyncio.run(main())
