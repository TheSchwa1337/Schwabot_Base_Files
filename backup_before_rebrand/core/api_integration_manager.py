"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Integration Manager
======================

Manages integration of external APIs (Glassnode, Whale Alert) with the Schwabot trading system.
Integrates with ZPE/ZBE thermal system and profit scheduler for enhanced decision making.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import time
import asyncio
import logging

# Import API handlers
try:
from core.api.handlers.glassnode import GlassnodeHandler
from core.api.handlers.whale_alert import WhaleAlertHandler
API_HANDLERS_AVAILABLE = True
except ImportError as e:
API_HANDLERS_AVAILABLE = False
logging.warning(f"API handlers not available: {e}")

# Import thermal and scheduler systems
try:
from core.thermal_strategy_router import ThermalStrategyRouter
from core.heartbeat_integration_manager import HeartbeatIntegrationManager
from core.profit_echo_cache import ProfitEchoCache
from core.drift_band_profiler import DriftBandProfiler
THERMAL_SYSTEMS_AVAILABLE = True
except ImportError as e:
THERMAL_SYSTEMS_AVAILABLE = False
logging.warning(f"Thermal systems not available: {e}")

logger = logging.getLogger(__name__)

class APIIntegrationManager:
"""Class for Schwabot trading functionality."""
"""
API Integration Manager for Schwabot trading system.

Coordinates external API data (Glassnode, Whale Alert) with internal
ZPE/ZBE thermal system and profit scheduler for enhanced decision making.
"""


def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize API integration manager."""
self.config = config or self._default_config()
self.logger = logging.getLogger(f"{__name__}.APIIntegrationManager")

# Initialize API handlers
self.glassnode_handler = None
self.whale_alert_handler = None

if API_HANDLERS_AVAILABLE:
try:
self.glassnode_handler = GlassnodeHandler(
api_key=self.config.get("glassnode_api_key", "demo-key")
)
self.whale_alert_handler = WhaleAlertHandler(
api_key=self.config.get("whale_alert_api_key", "demo-key")
)
self.logger.info("✅ API handlers initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing API handlers: {e}")

# Initialize thermal and scheduler systems
self.thermal_router = None
self.heartbeat_manager = None
self.profit_echo_cache = None
self.drift_profiler = None

if THERMAL_SYSTEMS_AVAILABLE:
try:
self.thermal_router = ThermalStrategyRouter()
self.heartbeat_manager = HeartbeatIntegrationManager()
self.profit_echo_cache = ProfitEchoCache()
self.drift_profiler = DriftBandProfiler()
self.logger.info("✅ Thermal systems initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing thermal systems: {e}")

# Integration state
self.is_running = False
self.last_integration_time = 0
self.integration_interval = self.config.get("integration_interval", 300)  # 5 minutes

# Data storage
self.glassnode_data = {}
self.whale_alert_data = {}
self.integrated_signals = {}

self.logger.info("✅ API Integration Manager initialized")

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
"enabled": True,
"integration_interval": 300,  # 5 minutes
"glassnode_api_key": "demo-key",
"whale_alert_api_key": "demo-key",
"thermal_integration": True,
"profit_scheduler_integration": True,
"debug": False,
}

async def start_integration(self) -> bool:
"""Start the API integration process."""
try:
if not self.config.get("enabled", True):
self.logger.warning("API integration is disabled in config")
return False

self.is_running = True
self.logger.info("🚀 Starting API integration process")

# Start integration loop
asyncio.create_task(self._integration_loop())

return True

except Exception as e:
self.logger.error(f"❌ Error starting API integration: {e}")
return False

async def stop_integration(self) -> bool:
"""Stop the API integration process."""
try:
self.is_running = False
self.logger.info("🛑 Stopping API integration process")
return True
except Exception as e:
self.logger.error(f"❌ Error stopping API integration: {e}")
return False

async def _integration_loop(self) -> None:
"""Main integration loop."""
while self.is_running:
try:
await self._perform_integration_cycle()
await asyncio.sleep(self.integration_interval)
except Exception as e:
self.logger.error(f"❌ Error in integration loop: {e}")
await asyncio.sleep(60)  # Wait 1 minute before retrying

async def _perform_integration_cycle(self) -> None:
"""Perform a complete integration cycle."""
cycle_start = time.time()
self.logger.info("🔄 Starting API integration cycle")

try:
# Step 1: Fetch API data
await self._fetch_api_data()

# Step 2: Process and integrate data
await self._process_integration_data()

# Step 3: Update thermal systems
if self.config.get("thermal_integration", True):
await self._update_thermal_systems()

# Step 4: Update profit scheduler
if self.config.get("profit_scheduler_integration", True):
await self._update_profit_scheduler()

# Step 5: Generate integrated signals
await self._generate_integrated_signals()

cycle_time = time.time() - cycle_start
self.last_integration_time = time.time()

self.logger.info(
f"✅ Integration cycle completed in {cycle_time:.2f}s")

except Exception as e:
self.logger.error(
f"❌ Error in integration cycle: {e}")

async def _fetch_api_data(self) -> None:
"""Fetch data from external APIs."""
try:
# Fetch Glassnode data
if self.glassnode_handler:
try:
raw_data = await self.glassnode_handler._fetch_raw()
parsed_data = await self.glassnode_handler._parse_raw(raw_data)
self.glassnode_handler._last_parsed_data = parsed_data
self.glassnode_data = parsed_data
self.logger.debug(
"✅ Glassnode data fetched successfully")
except Exception as e:
self.logger.error(
f"❌ Error fetching Glassnode data: {e}")

# Fetch Whale Alert data
if self.whale_alert_handler:
try:
raw_data = await self.whale_alert_handler._fetch_raw()
parsed_data = await self.whale_alert_handler._parse_raw(raw_data)
self.whale_alert_handler._last_parsed_data = parsed_data
self.whale_alert_data = parsed_data
self.logger.debug(
"✅ Whale Alert data fetched successfully")
except Exception as e:
self.logger.error(
f"❌ Error fetching Whale Alert data: {e}")

except Exception as e:
self.logger.error(
f"❌ Error fetching API data: {e}")

async def _process_integration_data(self) -> None:
"""Process and integrate API data."""
try:
# Process Glassnode data for thermal integration
if self.glassnode_data and self.glassnode_handler:
thermal_data = self.glassnode_handler.get_thermal_integration_data()
if thermal_data.get("thermal_ready", False):
self.logger.debug(
"✅ Glassnode thermal data processed")

# Process
# Whale
# Alert
# data
# for
# thermal
# integration
if self.whale_alert_data and self.whale_alert_handler:
thermal_data = self.whale_alert_handler.get_thermal_integration_data()
if thermal_data.get("thermal_ready", False):
self.logger.debug(
"✅ Whale Alert thermal data processed")

except Exception as e:
self.logger.error(
f"❌ Error processing integration data: {e}")

async def _update_thermal_systems(self) -> None:
"""Update ZPE/ZBE thermal systems with API data."""
try:
if not THERMAL_SYSTEMS_AVAILABLE:
return

# Update
# thermal
# strategy
# router
# with
# API
# data
if self.thermal_router:
thermal_signals = self._generate_thermal_signals()
if thermal_signals:
# Update thermal router with API-enhanced signals
self.logger.debug(
"✅ Thermal systems updated with API data")

# Update
# heartbeat
# integration
# manager
if self.heartbeat_manager:
heartbeat_data = self._generate_heartbeat_data()
if heartbeat_data:
# Update heartbeat manager with API data
self.logger.debug(
"✅ Heartbeat manager updated with API data")

except Exception as e:
self.logger.error(
f"❌ Error updating thermal systems: {e}")

async def _update_profit_scheduler(self) -> None:
"""Update profit scheduler with API data."""
try:
if not THERMAL_SYSTEMS_AVAILABLE:
return

# Update
# profit
# echo
# cache
# with
# whale
# activity
if self.profit_echo_cache and self.whale_alert_data:
whale_scheduler_data = self.whale_alert_handler.get_profit_scheduler_data()
if whale_scheduler_data.get("scheduler_ready", False):
# Update profit echo cache with whale signals
self.logger.debug(
"✅ Profit echo cache updated with whale data")

# Update
# drift
# band
# profiler
# with
# on-chain
# metrics
if self.drift_profiler and self.glassnode_data:
glassnode_scheduler_data = self.glassnode_handler.get_profit_scheduler_data()
if glassnode_scheduler_data.get("scheduler_ready", False):
# Update drift profiler with on-chain metrics
self.logger.debug(
"✅ Drift profiler updated with on-chain data")

except Exception as e:
self.logger.error(
f"❌ Error updating profit scheduler: {e}")

async def _generate_integrated_signals(self) -> None:
"""Generate integrated trading signals from API data."""
try:
integrated_signals = {
"timestamp": int(time.time()),
"glassnode_signals": {},
"whale_alert_signals": {},
"combined_signals": {},
"thermal_adjustments": {},
"profit_adjustments": {},
}

# Generate
# Glassnode
# signals
if self.glassnode_data:
integrated_signals["glassnode_signals"] = self._generate_glassnode_signals(
)

# Generate
# Whale
# Alert
# signals
if self.whale_alert_data:
integrated_signals["whale_alert_signals"] = self._generate_whale_alert_signals(
)

# Generate
# combined
# signals
integrated_signals["combined_signals"] = self._generate_combined_signals(
)

# Generate
# thermal
# adjustments
integrated_signals["thermal_adjustments"] = self._generate_thermal_adjustments(
)

# Generate
# profit
# adjustments
integrated_signals["profit_adjustments"] = self._generate_profit_adjustments(
)

self.integrated_signals = integrated_signals
self.logger.info(
"✅ Integrated signals generated successfully")

except Exception as e:
self.logger.error(
f"❌ Error generating integrated signals: {e}")

def _generate_glassnode_signals(self) -> Dict[str, Any]:
"""Generate trading signals from Glassnode data."""
try:
if not self.glassnode_data:
return {}

latest_values = self.glassnode_data.get(
"latest_values", {})
composite_scores = self.glassnode_data.get(
"composite_scores", {})

signals = {
"network_health": composite_scores.get("network_health", 50.0),
"valuation_health": composite_scores.get("valuation_health", 50.0),
"activity_level": composite_scores.get("activity_level", 50.0),
"mvrv_signal": self._interpret_mvrv(latest_values.get("mvrv", 1.0)),
"nvt_signal": self._interpret_nvt(latest_values.get("nvt", 50.0)),
"sopr_signal": self._interpret_sopr(latest_values.get("sopr", 1.0)),
"overall_signal": "neutral",
}

# Calculate
# overall
# signal
signals["overall_signal"] = self._calculate_overall_glassnode_signal(
signals)

return signals

except Exception as e:
self.logger.error(
f"Error generating Glassnode signals: {e}")
return {}

def _generate_whale_alert_signals(self) -> Dict[str, Any]:
"""Generate trading signals from Whale Alert data."""
try:
if not self.whale_alert_data:
return {}

whale_scores = self.whale_alert_data.get(
"whale_activity_scores", {})
market_impact = self.whale_alert_data.get(
"market_impact_analysis", {})

signals = {
"activity_level": whale_scores.get("activity_level", 0.0),
"volume_intensity": whale_scores.get("volume_intensity", 0.0),
"market_impact": market_impact.get("impact_score", 0.0),
"buying_pressure": market_impact.get("buying_pressure", 0.0),
"selling_pressure": market_impact.get("selling_pressure", 0.0),
"whale_signal": market_impact.get("risk_level", "low"),
"overall_signal": "neutral",
}

# Calculate
# overall
# signal
signals["overall_signal"] = self._calculate_overall_whale_signal(
signals)

return signals

except Exception as e:
self.logger.error(
f"Error generating Whale Alert signals: {e}")
return {}

def _generate_combined_signals(self) -> Dict[str, Any]:
"""Generate combined signals from both APIs."""
try:
glassnode_signals = self.integrated_signals.get(
"glassnode_signals", {})
whale_signals = self.integrated_signals.get(
"whale_alert_signals", {})

combined = {
"timestamp": int(time.time()),
"confidence_score": 0.0,
"action_signal": "hold",
"risk_level": "medium",
"volume_adjustment": 1.0,
"timing_adjustment": 1.0,
}

# Calculate
# confidence
# score
glassnode_confidence = self._signal_to_confidence(
glassnode_signals.get("overall_signal", "neutral"))
whale_confidence = self._signal_to_confidence(
whale_signals.get("overall_signal", "neutral"))

combined["confidence_score"] = (
glassnode_confidence + whale_confidence) / 2

# Determine
# action
# signal
combined["action_signal"] = self._determine_action_signal(
glassnode_signals, whale_signals)

# Determine
# risk
# level
combined["risk_level"] = self._determine_risk_level(
glassnode_signals, whale_signals)

# Calculate
# adjustments
combined["volume_adjustment"] = self._calculate_volume_adjustment(
combined)
combined["timing_adjustment"] = self._calculate_timing_adjustment(
combined)

return combined

except Exception as e:
self.logger.error(
f"Error generating combined signals: {e}")
return {}

def _interpret_mvrv(self, mvrv: float) -> str:
"""Interpret MVRV (Market Value to Realized Value) ratio."""
if mvrv < 0.8:
return "strong_buy"  # Heavily undervalued
elif mvrv < 1.2:
return "buy"  # Undervalued
elif mvrv < 2.0:
return "neutral"  # Fair value
elif mvrv < 3.0:
return "sell"  # Overvalued
else:
return "strong_sell"  # Heavily overvalued

def _interpret_nvt(self, nvt: float) -> str:
"""Interpret NVT (Network Value to Transactions) ratio."""
if nvt < 20:
# Very low NVT (good)
return "strong_buy"
elif nvt < 50:
return "buy"  # Low NVT
elif nvt < 100:
return "neutral"  # Normal NVT
else:
# High NVT (bad)
return "sell"

def _interpret_sopr(self, sopr: float) -> str:
"""Interpret SOPR (Spent Output Profit Ratio)."""
if sopr < 0.98:
# Strong selling pressure (good for buying)
return "strong_buy"
elif sopr < 1.0:
return "buy"  # Some selling pressure
elif sopr < 1.02:
return "neutral"  # Neutral
else:
# Strong buying pressure (good for selling)
return "sell"

def _calculate_overall_glassnode_signal(self, signals: Dict[str, Any]) -> str:
"""Calculate overall Glassnode signal."""
try:
# Weight the signals
mvrv_weight = 0.4
nvt_weight = 0.3
sopr_weight = 0.3

mvrv_score = self._signal_to_score(
signals.get("mvrv_signal", "neutral"))
nvt_score = self._signal_to_score(
signals.get("nvt_signal", "neutral"))
sopr_score = self._signal_to_score(
signals.get("sopr_signal", "neutral"))

weighted_score = (
mvrv_score * mvrv_weight +
nvt_score * nvt_weight +
sopr_score * sopr_weight
)

return self._score_to_signal(
weighted_score)

except Exception as e:
self.logger.error(
f"Error calculating overall Glassnode signal: {e}")
return "neutral"

def _calculate_overall_whale_signal(self, signals: Dict[str, Any]) -> str:
"""Calculate overall Whale Alert signal."""
try:
buying_pressure = signals.get(
"buying_pressure", 0.0)
selling_pressure = signals.get(
"selling_pressure", 0.0)
market_impact = signals.get(
"market_impact", 0.0)

if market_impact < 20:
return "neutral"  # Low impact

if buying_pressure > 70:
return "strong_buy"
elif buying_pressure > 60:
return "buy"
elif selling_pressure > 70:
return "strong_sell"
elif selling_pressure > 60:
return "sell"
else:
return "neutral"

except Exception as e:
self.logger.error(
f"Error calculating overall whale signal: {e}")
return "neutral"

def _signal_to_score(self, signal: str) -> float:
"""Convert signal string to numerical score."""
signal_map = {
"strong_buy": 1.0,
"buy": 0.5,
"neutral": 0.0,
"sell": -0.5,
"strong_sell": -1.0,
}
return signal_map.get(
signal, 0.0)

def _score_to_signal(self, score: float) -> str:
"""Convert numerical score to signal string."""
if score >= 0.7:
return "strong_buy"
elif score >= 0.3:
return "buy"
elif score >= -0.3:
return "neutral"
elif score >= -0.7:
return "sell"
else:
return "strong_sell"

def _signal_to_confidence(self, signal: str) -> float:
"""Convert signal to confidence score (0-1)."""
confidence_map = {
"strong_buy": 0.9,
"buy": 0.7,
"neutral": 0.5,
"sell": 0.7,
"strong_sell": 0.9,
}
return confidence_map.get(
signal, 0.5)

def _determine_action_signal(self, glassnode_signals: Dict, whale_signals: Dict) -> str:
"""Determine overall action signal from both APIs."""
try:
glassnode_signal = glassnode_signals.get(
"overall_signal", "neutral")
whale_signal = whale_signals.get(
"overall_signal", "neutral")

# Combine
# signals
# with
# weights
glassnode_weight = 0.6  # On-chain metrics more reliable
whale_weight = 0.4      # Whale activity more volatile

glassnode_score = self._signal_to_score(
glassnode_signal)
whale_score = self._signal_to_score(
whale_signal)

combined_score = (
glassnode_score * glassnode_weight +
whale_score * whale_weight
)

return self._score_to_signal(
combined_score)

except Exception as e:
self.logger.error(
f"Error determining action signal: {e}")
return "neutral"

def _determine_risk_level(self, glassnode_signals: Dict, whale_signals: Dict) -> str:
"""Determine overall risk level from both APIs."""
try:
whale_impact = whale_signals.get(
"market_impact", 0.0)
whale_risk = whale_signals.get(
"whale_signal", "low")

# High
# whale
# activity
# increases
# risk
if whale_impact > 80 or whale_risk == "critical":
return "high"
elif whale_impact > 60 or whale_risk == "high":
return "medium"
else:
return "low"

except Exception as e:
self.logger.error(
f"Error determining risk level: {e}")
return "medium"

def _calculate_volume_adjustment(self, combined_signals: Dict) -> float:
"""Calculate volume adjustment factor."""
try:
confidence = combined_signals.get(
"confidence_score", 0.5)
risk_level = combined_signals.get(
"risk_level", "medium")

# Base
# adjustment
# on
# confidence
base_adjustment = 0.5 + \
(confidence * 0.5)  # 0.5 to 1.0

# Adjust for risk
risk_multiplier = {
"low": 1.2,
"medium": 1.0,
"high": 0.8,
}.get(risk_level, 1.0)

return base_adjustment * risk_multiplier

except Exception as e:
self.logger.error(
f"Error calculating volume adjustment: {e}")
return 1.0

def _calculate_timing_adjustment(self, combined_signals: Dict) -> float:
"""Calculate timing adjustment factor."""
try:
confidence = combined_signals.get(
"confidence_score", 0.5)
action_signal = combined_signals.get(
"action_signal", "neutral")

# Higher
# confidence
# =
# faster
# execution
base_timing = 1.0 - \
(confidence * 0.3)  # 0.7 to 1.0

# Strong
# signals
# =
# faster
# execution
if "strong" in action_signal:
base_timing *= 0.8

# Minimum 0.5x timing
return max(0.5, base_timing)

except Exception as e:
self.logger.error(
f"Error calculating timing adjustment: {e}")
return 1.0

def _generate_thermal_signals(self) -> Dict[str, Any]:
"""Generate thermal signals for ZPE/ZBE integration."""
try:
return {
"api_enhanced": True,
"glassnode_thermal": self.glassnode_handler.get_thermal_integration_data() if self.glassnode_handler else {},
"whale_thermal": self.whale_alert_handler.get_thermal_integration_data() if self.whale_alert_handler else {},
}
except Exception as e:
self.logger.error(
f"Error generating thermal signals: {e}")
return {}

def _generate_heartbeat_data(self) -> Dict[str, Any]:
"""Generate heartbeat data for integration manager."""
try:
return {
"api_integration_active": True,
"last_integration_time": self.last_integration_time,
"integrated_signals": self.integrated_signals,
}
except Exception as e:
self.logger.error(
f"Error generating heartbeat data: {e}")
return {}

def _generate_thermal_adjustments(self) -> Dict[str, Any]:
"""Generate thermal adjustments for ZPE/ZBE system."""
try:
return {
"zpe_adjustment": 1.0,
"zbe_adjustment": 1.0,
"thermal_pressure": 0.5,
}
except Exception as e:
self.logger.error(
f"Error generating thermal adjustments: {e}")
return {}

def _generate_profit_adjustments(self) -> Dict[str, Any]:
"""Generate profit adjustments for scheduler."""
try:
combined_signals = self.integrated_signals.get(
"combined_signals", {})

return {
"volume_multiplier": combined_signals.get("volume_adjustment", 1.0),
"timing_multiplier": combined_signals.get("timing_adjustment", 1.0),
"confidence_boost": combined_signals.get("confidence_score", 0.5),
}
except Exception as e:
self.logger.error(
f"Error generating profit adjustments: {e}")
return {}

def get_integration_status(self) -> Dict[str, Any]:
"""Get current integration status."""
return {
"is_running": self.is_running,
"last_integration_time": self.last_integration_time,
"api_handlers_available": API_HANDLERS_AVAILABLE,
"thermal_systems_available": THERMAL_SYSTEMS_AVAILABLE,
"glassnode_data_available": bool(self.glassnode_data),
"whale_alert_data_available": bool(self.whale_alert_data),
"integrated_signals_available": bool(self.integrated_signals),
}

def get_latest_signals(self) -> Dict[str, Any]:
"""Get latest integrated signals."""
return self.integrated_signals.copy() if self.integrated_signals else {}
