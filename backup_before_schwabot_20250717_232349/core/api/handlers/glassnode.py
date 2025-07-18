"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glassnode API Handler
====================

Fetches on-chain cryptocurrency metrics from Glassnode API.
Provides network health, market valuation, and activity metrics.
Integrates with ZPE/ZBE thermal system and profit scheduler.
"""

from typing import Any, Dict, List, Optional
import time
import asyncio
import logging

try:
import aiohttp
except ImportError:
aiohttp = None

try:
import requests
except ImportError:
requests = None

from .base_handler import BaseAPIHandler

logger = logging.getLogger(__name__)

# Glassnode API configuration
BASE_URL = "https://api.glassnode.com/v1/metrics"

class GlassnodeHandler(BaseAPIHandler):
"""Class for Schwabot trading functionality."""
"""Glassnode API handler for on-chain cryptocurrency metrics."""

NAME = "glassnode"
CACHE_SUBDIR = "onchain_data"
REFRESH_INTERVAL = 900  # 15-minute updates for on-chain data


def __init__(self, api_key: str = None, cache_root: str = "flask/feeds") -> None:
"""Initialize Glassnode handler with API key and cache configuration."""
super().__init__()
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
trend_percent = (
(last_value - first_value) / first_value) * 100
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
logger.error(
f"{self.NAME}: failed to parse Glassnode data – {exc}")
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
hash_rate = latest_values.get(
"hash_rate_mean", 0)
active_addresses = latest_values.get(
"active_count", 0)
transaction_count = latest_values.get(
"count", 0)

# Normalize and combine (simplified scoring)
network_score = min(
100,
(
(hash_rate / 1e18 * 20)  # Hash rate component
+ (active_addresses / 1000000 * 30)  # Active addresses component
+ (transaction_count / 300000 * 50)  # Transaction count component
),
)
scores["network_health"] = max(
0, network_score)

# Market valuation score
mvrv = latest_values.get("mvrv", 1.0)
nvt = latest_values.get("nvt", 50.0)
sopr = latest_values.get("sopr", 1.0)

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

scores["valuation_health"] = min(
100, max(0, valuation_score))

# Activity score
price_change = latest_values.get(
"price_usd_close", 0)
activity_score = min(
100, max(0, 50 + (transaction_count / 200000 * 50)))
scores["activity_level"] = activity_score

except Exception as e:
logger.error(
f"Error calculating composite scores: {e}")
scores = {
"network_health": 50.0,
"valuation_health": 50.0,
"activity_level": 50.0,
}

return scores

async def _get_session(self) -> aiohttp.ClientSession:
"""Get aiohttp session for API calls."""
if not hasattr(self, '_session') or self._session.closed:
self._session = aiohttp.ClientSession()
return self._session

def get_thermal_integration_data(self) -> Dict[str, Any]:
"""Get data formatted for ZPE/ZBE thermal integration."""
try:
if not hasattr(
self, '_last_parsed_data'):
return {
"thermal_ready": False, "error": "No data available"}

data = self._last_parsed_data
latest_values = data.get(
"latest_values", {})
composite_scores = data.get(
"composite_scores", {})

# Format for thermal system integration
thermal_data = {
"thermal_ready": True,
"network_health": composite_scores.get("network_health", 50.0),
"valuation_health": composite_scores.get("valuation_health", 50.0),
"activity_level": composite_scores.get("activity_level", 50.0),
"mvrv": latest_values.get("mvrv", 1.0),
"nvt": latest_values.get("nvt", 50.0),
"sopr": latest_values.get("sopr", 1.0),
"hash_rate": latest_values.get("hash_rate_mean", 0),
"active_addresses": latest_values.get("active_count", 0),
"transaction_count": latest_values.get("count", 0),
"timestamp": data.get("timestamp", int(time.time())),
}

return thermal_data

except Exception as e:
logger.error(
f"Error getting thermal integration data: {e}")
return {
"thermal_ready": False, "error": str(e)}

def get_profit_scheduler_data(self) -> Dict[str, Any]:
"""Get data formatted for profit scheduler integration."""
try:
if not hasattr(
self, '_last_parsed_data'):
return {
"scheduler_ready": False, "error": "No data available"}

data = self._last_parsed_data
latest_values = data.get(
"latest_values", {})
trends = data.get(
"trends", {})

# Format for profit scheduler
scheduler_data = {
"scheduler_ready": True,
"market_cap": latest_values.get("marketcap_usd", 0),
"price": latest_values.get("price_usd_close", 0),
"mvrv_trend": trends.get("mvrv", 0),
"nvt_trend": trends.get("nvt", 0),
"sopr_trend": trends.get("sopr", 0),
"network_activity": latest_values.get("count", 0),
"valuation_score": self._calculate_valuation_score(latest_values),
"timestamp": data.get("timestamp", int(time.time())),
}

return scheduler_data

except Exception as e:
logger.error(
f"Error getting profit scheduler data: {e}")
return {
"scheduler_ready": False, "error": str(e)}

def _calculate_valuation_score(self, latest_values: Dict[str, float]) -> float:
"""Calculate valuation score for profit scheduling."""
try:
mvrv = latest_values.get(
"mvrv", 1.0)
nvt = latest_values.get(
"nvt", 50.0)
sopr = latest_values.get(
"sopr", 1.0)

# MVRV
# scoring
# (0-100)
if mvrv < 0.8:
mvrv_score = 100  # Heavily undervalued
elif mvrv < 1.2:
mvrv_score = 80   # Undervalued
elif mvrv < 2.0:
mvrv_score = 50   # Fair value
elif mvrv < 3.0:
mvrv_score = 20   # Overvalued
else:
mvrv_score = 0    # Heavily overvalued

# NVT
# scoring
# (0-100)
if nvt < 20:
# Very low NVT (good)
nvt_score = 100
elif nvt < 50:
nvt_score = 80    # Low NVT
elif nvt < 100:
nvt_score = 50    # Normal NVT
else:
# High NVT (bad)
nvt_score = 20

# SOPR
# scoring
# (0-100)
if sopr < 0.98:
sopr_score = 100  # Strong selling pressure
elif sopr < 1.0:
sopr_score = 80   # Some selling pressure
elif sopr < 1.02:
sopr_score = 50   # Neutral
else:
sopr_score = 20   # Strong buying pressure

# Weighted
# average
valuation_score = (
mvrv_score * 0.4 + nvt_score * 0.3 + sopr_score * 0.3)
return max(
0, min(100, valuation_score))

except Exception as e:
logger.error(
f"Error calculating valuation score: {e}")
return 50.0  # Neutral default
