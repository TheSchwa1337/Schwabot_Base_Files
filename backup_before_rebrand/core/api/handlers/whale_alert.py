"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whale Alert API Handler
======================

Fetches large cryptocurrency transactions from Whale Alert API.
Provides whale activity monitoring and market impact analysis.
Integrates with ZPE/ZBE thermal system and profit scheduler.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
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

# Whale Alert API configuration
BASE_URL = "https://api.whale-alert.io/v1"

class WhaleAlertHandler(BaseAPIHandler):
"""Class for Schwabot trading functionality."""
"""Whale Alert API handler for large cryptocurrency transactions."""

NAME = "whale_alert"
CACHE_SUBDIR = "whale_data"
REFRESH_INTERVAL = 300  # 5-minute updates for whale activity


def __init__(self, api_key: str = None, cache_root: str = "flask/feeds") -> None:
"""Initialize Whale Alert handler with API key and cache configuration."""
super().__init__()
self.api_key = api_key or "demo-key"

# Supported blockchains
self.blockchains = ["bitcoin", "ethereum", "tether"]

# Minimum transaction value (USD) to track
self.min_value = 500000  # $500K minimum

# Time window for analysis (hours)
self.analysis_window = 24

async def _fetch_raw(self) -> Any:
"""Fetch raw whale transaction data from Whale Alert API."""
all_data = {}

# Calculate time window
end_time = int(time.time())
start_time = end_time - (self.analysis_window * 3600)  # 24 hours back

for blockchain in self.blockchains:
params = {
"api_key": self.api_key,
"start": start_time,
"end": end_time,
"min_value": self.min_value,
"cursor": "",  # For pagination
}

try:
if aiohttp:
session = await self._get_session()
async with session.get(f"{BASE_URL}/transactions", params=params) as resp:
if resp.status == 401:
logger.warning(f"Whale Alert API key invalid for {blockchain}")
all_data[blockchain] = []
continue
elif resp.status != 200:
logger.warning(f"Whale Alert API error {resp.status} for {blockchain}")
all_data[blockchain] = []
continue
data = await resp.json()
all_data[blockchain] = data.get("transactions", [])
elif requests:
loop = asyncio.get_running_loop()
response = await loop.run_in_executor(
None,
lambda: requests.get(f"{BASE_URL}/transactions", params=params, timeout=15),
)
if response.status_code == 401:
logger.warning(f"Whale Alert API key invalid for {blockchain}")
all_data[blockchain] = []
continue
elif response.status_code != 200:
logger.warning(f"Whale Alert API error {response.status_code} for {blockchain}")
all_data[blockchain] = []
continue
data = response.json()
all_data[blockchain] = data.get("transactions", [])
else:
raise RuntimeError("Neither aiohttp nor requests is available for HTTP calls")

# Small delay between requests to respect rate limits
await asyncio.sleep(0.5)

except Exception as e:
logger.error(f"Failed to fetch Whale Alert data for {blockchain}: {e}")
all_data[blockchain] = []

return all_data

async def _parse_raw(self, raw: Any) -> Dict[str, Any]:
"""Parse whale transaction data into normalized format."""
try:
parsed_data = {
"timestamp": int(time.time()),
"transactions": [],
"blockchain_stats": {},
"whale_activity_scores": {},
"market_impact_analysis": {},
}

total_transactions = []

# Process transactions from all blockchains
for blockchain, transactions in raw.items():
if not transactions or not isinstance(transactions, list):
continue

blockchain_transactions = []

for tx in transactions:
if not isinstance(tx, dict):
continue

# Extract transaction data
tx_data = {
"blockchain": blockchain,
"hash": tx.get("hash", ""),
"timestamp": tx.get("timestamp", 0),
"amount": float(tx.get("amount", 0)),
"amount_usd": float(tx.get("amount_usd", 0)),
"from": tx.get("from", {}),
"to": tx.get("to", {}),
"symbol": tx.get("symbol", ""),
"transaction_type": tx.get("transaction_type", ""),
}

blockchain_transactions.append(tx_data)
total_transactions.append(tx_data)

# Calculate blockchain-specific stats
if blockchain_transactions:
parsed_data["blockchain_stats"][blockchain] = self._calculate_blockchain_stats(
blockchain_transactions
)

# Store all transactions
parsed_data["transactions"] = total_transactions

# Calculate overall whale activity scores
if total_transactions:
parsed_data["whale_activity_scores"] = self._calculate_whale_activity_scores(
total_transactions
)

# Calculate market impact analysis
parsed_data["market_impact_analysis"] = self._calculate_market_impact(
total_transactions
)

return parsed_data

except Exception as exc:
logger.error(
f"{self.NAME}: failed to parse Whale Alert data – {exc}")
return {
"timestamp": int(time.time()),
"transactions": [],
"blockchain_stats": {},
"whale_activity_scores": {},
"market_impact_analysis": {},
}

def _calculate_blockchain_stats(self, transactions: List[Dict]) -> Dict[str, Any]:
"""Calculate statistics for a specific blockchain."""
try:
if not transactions:
return {}

total_amount = sum(tx["amount_usd"] for tx in transactions)
total_count = len(transactions)
avg_amount = total_amount / total_count if total_count > 0 else 0

# Categorize by transaction type
type_counts = {}
type_amounts = {}

for tx in transactions:
tx_type = tx.get("transaction_type", "unknown")
type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
type_amounts[tx_type] = type_amounts.get(
tx_type, 0) + tx["amount_usd"]

return {
"total_transactions": total_count,
"total_volume_usd": total_amount,
"average_transaction_usd": avg_amount,
"transaction_types": type_counts,
"volume_by_type": type_amounts,
"largest_transaction": max(tx["amount_usd"] for tx in transactions),
"smallest_transaction": min(tx["amount_usd"] for tx in transactions),
}

except Exception as e:
logger.error(f"Error calculating blockchain stats: {e}")
return {}

def _calculate_whale_activity_scores(self, transactions: List[Dict]) -> Dict[str, float]:
"""Calculate whale activity scores (0-100 scale)."""
try:
if not transactions:
return {
"activity_level": 0.0,
"volume_intensity": 0.0,
"frequency_score": 0.0,
"size_score": 0.0,
}

# Activity level based on transaction count
tx_count = len(transactions)
activity_level = min(
100, (tx_count / 50) * 100)  # Normalize to 50 transactions = 100%

# Volume intensity based on total volume
total_volume = sum(tx["amount_usd"]
for tx in transactions)
volume_intensity = min(
100, (total_volume / 1000000000) * 100)  # Normalize to $1B = 100%

# Frequency score (transactions per hour)
time_window_hours = self.analysis_window
transactions_per_hour = tx_count / time_window_hours
frequency_score = min(
100, (transactions_per_hour / 10) * 100)  # Normalize to 10 tx/hour = 100%

# Size score based on average transaction size
avg_size = total_volume / tx_count if tx_count > 0 else 0
size_score = min(
100, (avg_size / 10000000) * 100)  # Normalize to $10M avg = 100%

return {
"activity_level": activity_level,
"volume_intensity": volume_intensity,
"frequency_score": frequency_score,
"size_score": size_score,
}

except Exception as e:
logger.error(
f"Error calculating whale activity scores: {e}")
return {
"activity_level": 0.0,
"volume_intensity": 0.0,
"frequency_score": 0.0,
"size_score": 0.0,
}

def _calculate_market_impact(self, transactions: List[Dict]) -> Dict[str, Any]:
"""Calculate market impact analysis."""
try:
if not transactions:
return {
"impact_score": 0.0,
"buying_pressure": 0.0,
"selling_pressure": 0.0,
"net_flow": 0.0,
"risk_level": "low",
}

# Analyze transaction types for market impact
buying_volume = 0
selling_volume = 0

for tx in transactions:
tx_type = tx.get("transaction_type", "").lower()
volume = tx["amount_usd"]

if "exchange" in tx_type or "sell" in tx_type:
selling_volume += volume
elif "buy" in tx_type or "purchase" in tx_type:
buying_volume += volume

total_volume = buying_volume + selling_volume

# Calculate pressure scores
buying_pressure = (
buying_volume / total_volume * 100) if total_volume > 0 else 0
selling_pressure = (
selling_volume / total_volume * 100) if total_volume > 0 else 0

# Net flow (positive = net buying,
# negative = net selling)
net_flow = buying_volume - selling_volume

# Overall impact score
impact_score = min(
100, (total_volume / 500000000) * 100)  # Normalize to $500M = 100%

# Risk level assessment
if impact_score > 80:
risk_level = "critical"
elif impact_score > 60:
risk_level = "high"
elif impact_score > 40:
risk_level = "medium"
elif impact_score > 20:
risk_level = "low"
else:
risk_level = "minimal"

return {
"impact_score": impact_score,
"buying_pressure": buying_pressure,
"selling_pressure": selling_pressure,
"net_flow": net_flow,
"risk_level": risk_level,
"total_volume": total_volume,
}

except Exception as e:
logger.error(
f"Error calculating market impact: {e}")
return {
"impact_score": 0.0,
"buying_pressure": 0.0,
"selling_pressure": 0.0,
"net_flow": 0.0,
"risk_level": "low",
}

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
whale_scores = data.get(
"whale_activity_scores", {})
market_impact = data.get(
"market_impact_analysis", {})

# Format for thermal system integration
thermal_data = {
"thermal_ready": True,
"whale_activity_level": whale_scores.get("activity_level", 0.0),
"volume_intensity": whale_scores.get("volume_intensity", 0.0),
"frequency_score": whale_scores.get("frequency_score", 0.0),
"market_impact_score": market_impact.get("impact_score", 0.0),
"buying_pressure": market_impact.get("buying_pressure", 0.0),
"selling_pressure": market_impact.get("selling_pressure", 0.0),
"risk_level": market_impact.get("risk_level", "low"),
"total_whale_volume": market_impact.get("total_volume", 0),
"transaction_count": len(data.get("transactions", [])),
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
whale_scores = data.get(
"whale_activity_scores", {})
market_impact = data.get(
"market_impact_analysis", {})

# Format for profit scheduler
scheduler_data = {
"scheduler_ready": True,
"whale_activity_score": whale_scores.get("activity_level", 0.0),
"volume_intensity": whale_scores.get("volume_intensity", 0.0),
"market_impact": market_impact.get("impact_score", 0.0),
"buying_pressure": market_impact.get("buying_pressure", 0.0),
"selling_pressure": market_impact.get("selling_pressure", 0.0),
"net_flow": market_impact.get("net_flow", 0.0),
"risk_level": market_impact.get("risk_level", "low"),
"whale_signal": self._calculate_whale_signal(market_impact),
"timestamp": data.get("timestamp", int(time.time())),
}

return scheduler_data

except Exception as e:
logger.error(
f"Error getting profit scheduler data: {e}")
return {
"scheduler_ready": False, "error": str(e)}

def _calculate_whale_signal(self, market_impact: Dict[str, Any]) -> str:
"""Calculate whale signal for profit scheduling."""
try:
buying_pressure = market_impact.get(
"buying_pressure", 0.0)
selling_pressure = market_impact.get(
"selling_pressure", 0.0)
impact_score = market_impact.get(
"impact_score", 0.0)

# Determine
# whale
# signal
# based
# on
# pressure
# and
# impact
if impact_score < 20:
return "neutral"  # Low impact

if buying_pressure > 70:
return "strong_buy"  # Strong buying pressure
elif buying_pressure > 60:
return "buy"  # Moderate buying pressure
elif selling_pressure > 70:
return "strong_sell"  # Strong selling pressure
elif selling_pressure > 60:
return "sell"  # Moderate selling pressure
else:
return "neutral"  # Balanced pressure

except Exception as e:
logger.error(
f"Error calculating whale signal: {e}")
return "neutral"
