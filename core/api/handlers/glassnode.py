from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

import aiohttp
import requests

from .base_handler import BaseAPIHandler

"""Glassnode API Handler







Fetches on-chain cryptocurrency metrics from Glassnode API.



Provides network health, market valuation, and activity metrics.



"""


try:

    import aiohttp


except ImportError:

    aiohttp = None


try:

    import requests


except ImportError:

    requests = None


logger = logging.getLogger(__name__)


# Glassnode API configuration


BASE_URL = "https://api.glassnode.com/v1/metrics"


class GlassnodeHandler(BaseAPIHandler):
    """Glassnode API handler for on-chain cryptocurrency metrics."""

    NAME = "glassnode"

    CACHE_SUBDIR = "onchain_data"

    REFRESH_INTERVAL = 900  # 15-minute updates for on-chain data

    def __init__(self, api_key: str = None, cache_root: str = "flask/feeds"):
        """Initializes the Glassnode handler with an API key and cache configuration."""

        super().__init__(cache_root)

        self.api_key = api_key or "demo-key"

        self.asset = "BTC"  # Default to Bitcoin

        # Key metrics to track

        self.metrics = [
            "market/marketcap_usd",
            "market/price_usd_close",
            "transactions/count",
            "addresses/active_count",
            "network/hash_rate_mean",
            "market/mvrv",
            "indicators/nvt",
            "indicators/sopr",
            "supply/current",
            "mining/difficulty_latest",
        ]

    async def _fetch_raw(self) -> Any:
        """Fetch raw on-chain metrics from Glassnode API."""

        all_data = {}

        for metric in self.metrics:

            params = {
                "a": self.asset,
                "api_key": self.api_key,
                "s": int(time.time() - 86400),  # Last 24 hours
                "u": int(time.time()),
                "i": "1h",  # 1-hour intervals
            }

            try:

                if aiohttp:

                    session = await self._get_session()

                    async with session.get(f"{BASE_URL}/{metric}", params=params) as resp:

                        if resp.status == 401:

                            logger.warning(f"Glassnode API key invalid for metric {metric}")

                            all_data[metric] = []

                            continue

                        elif resp.status != 200:

                            logger.warning(f"Glassnode API error {resp.status} for metric {metric}")

                            all_data[metric] = []

                            continue

                        data = await resp.json()

                        all_data[metric] = data

                elif requests:

                    loop = asyncio.get_running_loop()

                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(f"{BASE_URL}/{metric}", params=params, timeout=15),
                    )

                    if response.status_code == 401:

                        logger.warning(f"Glassnode API key invalid for metric {metric}")

                        all_data[metric] = []

                        continue

                    elif response.status_code != 200:

                        logger.warning(
                            f"Glassnode API error {response.status_code} for metric {metric}"
                        )

                        all_data[metric] = []

                        continue

                    all_data[metric] = response.json()

                else:

                    raise RuntimeError("Neither aiohttp nor requests is available for HTTP calls")

                # Small delay between requests to respect rate limits

                await asyncio.sleep(0.2)

            except Exception as e:

                logger.error(f"Failed to fetch Glassnode metric {metric}: {e}")

                all_data[metric] = []

        return all_data

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
        """Parse Glassnode metrics into normalized format."""

        try:

            parsed_data = {
                "asset": self.asset,
                "timestamp": int(time.time()),
                "metrics": {},
                "latest_values": {},
                "trends": {},
            }

            for metric_path, data_points in raw.items():

                if not data_points or not isinstance(data_points, list):

                    continue

                metric_name = metric_path.split("/")[-1]

                # Process data points

                values = []

                timestamps = []

                for point in data_points:

                    if isinstance(point, dict) and "t" in point and "v" in point:

                        timestamps.append(point["t"])

                        value = point["v"]

                        if value is not None:

                            values.append(float(value))

                        else:

                            values.append(0.0)

                if not values:

                    continue

                # Store processed data

                parsed_data["metrics"][metric_name] = {
                    "values": values,
                    "timestamps": timestamps,
                    "count": len(values),
                }

                # Latest value

                parsed_data["latest_values"][metric_name] = values[-1] if values else 0.0

                # Calculate trend (24h change)

                if len(values) >= 2:

                    first_value = values[0]

                    last_value = values[-1]

                    if first_value != 0:

                        trend_percent = ((last_value - first_value) / first_value) * 100

                    else:

                        trend_percent = 0.0

                    parsed_data["trends"][metric_name] = trend_percent

                else:

                    parsed_data["trends"][metric_name] = 0.0

            # Calculate composite scores

            parsed_data["composite_scores"] = self._calculate_composite_scores(
                parsed_data["latest_values"]
            )

            return parsed_data

        except Exception as exc:

            logger.error("%s: failed to parse Glassnode data - %s", self.NAME, exc)

            return {
                "asset": self.asset,
                "metrics": {},
                "latest_values": {},
                "trends": {},
            }

    def _calculate_composite_scores(self, latest_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite scores from multiple metrics."""

        scores = {}

        try:

            # Network health score (0-100)

            hash_rate = latest_values.get("hash_rate_mean", 0)

            active_addresses = latest_values.get("active_count", 0)

            transaction_count = latest_values.get("count", 0)

            # Normalize and combine (simplified scoring)

            network_score = min(
                100,
                (
                    (hash_rate / 1e18 * 20)  # Hash rate component
                    + (active_addresses / 1000000 * 30)  # Active addresses component
                    + (transaction_count / 300000 * 50)  # Transaction count component
                ),
            )

            scores["network_health"] = max(0, network_score)

            # Market valuation score

            mvrv = latest_values.get("mvrv", 1.0)

            nvt = latest_values.get("nvt", 50.0)

            latest_values.get("sopr", 1.0)

            # Valuation assessment (simplified)

            if mvrv > 3.0:

                valuation_score = 20  # Overvalued

            elif mvrv > 1.5:

                valuation_score = 50  # Fair value

            elif mvrv > 0.8:

                valuation_score = 80  # Undervalued

            else:

                valuation_score = 100  # Heavily undervalued

            # Adjust based on NVT

            if nvt > 100:

                valuation_score *= 0.8  # Reduce score for high NVT

            elif nvt < 20:

                valuation_score *= 1.2  # Increase score for low NVT

            scores["valuation_health"] = min(100, max(0, valuation_score))

            # Activity score

            activity_score = min(
                100,
                (
                    (transaction_count / 300000 * 40)  # Transaction count
                    + (active_addresses / 1000000 * 60)  # Active addresses
                ),
            )

            scores["activity_score"] = max(0, activity_score)

            # Combine into an overall score

            overall_score = (
                scores["network_health"] + scores["valuation_health"] + scores["activity_score"]
            ) / 3

            scores["overall_score"] = round(overall_score, 2)

        except Exception as e:

            logger.error(f"Error calculating composite scores: {e}")

            scores = {
                "network_health": 0.0,
                "valuation_health": 0.0,
                "activity_score": 0.0,
                "overall_score": 0.0,
            }

        return scores


# Example Usage


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    handler = GlassnodeHandler(api_key="YOUR_GLASSNODE_API_KEY")  # Replace with your actual API key

    async def main():

        data = await handler.get_data(force_refresh=True)

        if data:

            logger.info("Successfully fetched and parsed Glassnode data.")

            logger.info(json.dumps(data, indent=2))

        else:

            logger.error("Failed to fetch Glassnode data.")

    asyncio.run(main())
