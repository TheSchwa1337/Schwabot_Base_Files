"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile Router for Multi-Profile Coinbase Trading
================================================
Routes and manages multiple Coinbase API profiles with integration
to the existing Schwabot trading system infrastructure.

Mathematical Core:
∀ t: H₁(t) ≠ H₂(t) ∨ A₁ ≠ A₂
Strategy_Profile(t, Pᵢ) = ƒ(Hashₜᵢ, Assetsᵢ, Holdingsᵢ, Profit_Zonesᵢ)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Import multi-profile components
from .api.multi_profile_coinbase_manager import MultiProfileCoinbaseManager
from .strategy_mapper import StrategyMapper

# Import existing Schwabot components
try:
from .live_trading_system import LiveTradingSystem
from .portfolio_tracker import EnhancedPortfolioTracker
from .risk_manager_enhanced import RiskManager
SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError:
SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProfileRouter:
"""Class for Schwabot trading functionality."""
"""
Profile Router for Multi-Profile Trading

Integrates multiple Coinbase API profiles with the existing Schwabot system:
- Routes strategies to appropriate profiles
- Manages cross-profile synchronization
- Integrates with existing trading infrastructure
- Provides unified interface for multi-profile operations
"""


def __init__(self, config_path: str = "config/coinbase_profiles.yaml") -> None:
"""Initialize the profile router."""
self.config_path = config_path
self.logger = logging.getLogger(__name__)

# Initialize multi-profile components
self.multi_profile_manager = MultiProfileCoinbaseManager(config_path)
self.strategy_mapper = StrategyMapper()

# Integration with existing Schwabot components
self.live_trading_system = None
self.portfolio_tracker = None
self.risk_manager = None

if SCHWABOT_COMPONENTS_AVAILABLE:
try:
self.live_trading_system = LiveTradingSystem()
self.portfolio_tracker = EnhancedPortfolioTracker({})
self.risk_manager = RiskManager()
self.logger.info("✅ Schwabot components integrated")
except Exception as e:
self.logger.warning(f"⚠️ Schwabot components not available: {e}")

# Profile routing state
self.active_profiles = {}
self.profile_routes = {}
self.cross_profile_arbitration_enabled = True
self.unified_interface_enabled = True

# Performance tracking
self.total_routed_trades = 0
self.successful_routes = 0
self.failed_routes = 0
self.arbitration_opportunities = 0

# System state
self.initialized = False
self.running = False

self.logger.info("Profile Router initialized")

async def initialize(self) -> bool:
"""Initialize the profile router system."""
try:
self.logger.info("🔄 Initializing Profile Router...")

# Initialize multi-profile manager
if not await self.multi_profile_manager.initialize_profiles():
self.logger.error("❌ Failed to initialize multi-profile manager")
return False

# Register profiles with strategy mapper
profiles_config = self.multi_profile_manager.config.get('profiles', {})
for profile_id, profile_config in profiles_config.items():
if profile_config.get('enabled', False):
self.strategy_mapper.register_profile(profile_id, profile_config)
self.active_profiles[profile_id] = profile_config

# Initialize existing Schwabot components if available
if SCHWABOT_COMPONENTS_AVAILABLE and self.live_trading_system:
try:
await self.live_trading_system.initialize()
self.logger.info("✅ Live trading system integrated")
except Exception as e:
self.logger.warning(f"⚠️ Live trading system integration failed: {e}")

self.initialized = True
self.logger.info(f"✅ Profile Router initialized with {len(self.active_profiles)} active profiles")
return True

except Exception as e:
self.logger.error(f"Failed to initialize Profile Router: {e}")
return False

async def start_trading(self) -> bool:
"""Start multi-profile trading."""
try:
if not self.initialized:
raise RuntimeError("Profile Router not initialized")

self.running = True
self.logger.info("🚀 Starting multi-profile trading...")

# Start multi-profile manager
trading_task = asyncio.create_task(self.multi_profile_manager.start_trading())

# Start profile routing loop
routing_task = asyncio.create_task(self._profile_routing_loop())

# Start integration loop
integration_task = asyncio.create_task(self._integration_loop())

# Wait for all tasks
await asyncio.gather(trading_task, routing_task, integration_task, return_exceptions=True)

return True

except Exception as e:
self.logger.error(f"Error starting trading: {e}")
return False

async def _profile_routing_loop(self):
"""Main profile routing loop."""
try:
self.logger.info("🔄 Starting profile routing loop...")

while self.running:
# Route strategies to profiles
await self._route_strategies_to_profiles()

# Check for cross-profile opportunities
if self.cross_profile_arbitration_enabled:
await self._check_cross_profile_opportunities()

# Update routing metrics
await self._update_routing_metrics()

# Sleep between routing cycles
await asyncio.sleep(30)  # 30-second routing cycle

except Exception as e:
self.logger.error(f"Error in profile routing loop: {e}")

async def _route_strategies_to_profiles(self):
"""Route strategies to appropriate profiles."""
try:
for profile_id, profile_config in self.active_profiles.items():
# Get current hash state
hash_state = self.multi_profile_manager.profile_hashes.get(profile_id)
if not hash_state:
continue

# Generate strategy for profile
strategy_matrix = await self.strategy_mapper.generate_profile_strategy(
profile_id, profile_config, hash_state.current_hash
)

if strategy_matrix:
# Check strategy uniqueness
is_unique = await self.strategy_mapper.check_strategy_uniqueness(profile_id)

if is_unique:
# Route strategy to profile
await self._execute_profile_strategy(profile_id, strategy_matrix)
self.successful_routes += 1
else:
# Strategy duplication detected, skip execution
self.failed_routes += 1
self.logger.warning(f"⚠️ Strategy duplication detected for profile {profile_id}")

self.total_routed_trades += 1

except Exception as e:
self.logger.error(f"Error routing strategies to profiles: {e}")

async def _execute_profile_strategy(self, profile_id: str, strategy_matrix):
"""Execute strategy for specific profile."""
try:
# Check if profile is active
if self.multi_profile_manager.profile_states.get(
profile_id) != self.multi_profile_manager.profile_states.get(profile_id, None):
return

# Get profile API
profile_api = self.multi_profile_manager.profiles.get(profile_id)
if not profile_api:
return

# Execute strategy through multi-profile manager
strategy_data = {
'profile_id': profile_id,
'strategy_name': strategy_matrix.strategy_type.value,
'assets': strategy_matrix.assets,
'confidence': strategy_matrix.confidence,
'signal_strength': strategy_matrix.signal_strength,
'hash_state': strategy_matrix.hash_state,
'timestamp': strategy_matrix.timestamp,
'profile_hash': strategy_matrix.hash_state
}

await self.multi_profile_manager._execute_profile_strategy(profile_id, strategy_data)

# Update strategy performance
await self.strategy_mapper.update_strategy_performance(
profile_id, strategy_matrix.strategy_type.value, strategy_matrix.confidence
)

self.logger.debug(f"✅ Strategy executed for profile {profile_id}: {strategy_matrix.strategy_type.value}")

except Exception as e:
self.logger.error(f"Error executing strategy for profile {profile_id}: {e}")

async def _check_cross_profile_opportunities(self):
"""Check for cross-profile arbitrage opportunities."""
try:
# Get arbitration history
arbitration_history = self.multi_profile_manager.get_arbitration_history()

if arbitration_history:
latest_arbitration = arbitration_history[-1]

if latest_arbitration.get('opportunity_detected', False):
self.arbitration_opportunities += 1

# Execute arbitrage strategy
await self._execute_arbitrage_strategy(latest_arbitration)

except Exception as e:
self.logger.error(f"Error checking cross-profile opportunities: {e}")

async def _execute_arbitrage_strategy(self, arbitration_data: Dict[str, Any]):
"""Execute arbitrage strategy between profiles."""
try:
profile_a = arbitration_data.get('profile_a')
profile_b = arbitration_data.get('profile_b')
score = arbitration_data.get('score', 0)

if score > 0.9:  # High arbitrage opportunity
self.logger.info(f"💰 High arbitrage opportunity detected between {profile_a} and {profile_b} (score: {score:.3f})")

# Execute arbitrage through multi-profile manager
await self.multi_profile_manager._execute_arbitrage_strategy(profile_a, profile_b, score)

except Exception as e:
self.logger.error(f"Error executing arbitrage strategy: {e}")

async def _integration_loop(self):
"""Integration loop with existing Schwabot components."""
try:
self.logger.info("🔄 Starting integration loop...")

while self.running:
# Integrate with portfolio tracker if available
if self.portfolio_tracker:
await self._integrate_portfolio_data()

# Integrate with risk manager if available
if self.risk_manager:
await self._integrate_risk_management()

# Sleep between integration cycles
await asyncio.sleep(60)  # 1-minute integration cycle

except Exception as e:
self.logger.error(f"Error in integration loop: {e}")

async def _integrate_portfolio_data(self):
"""Integrate portfolio data with multi-profile system."""
try:
# Get portfolio data from existing system
portfolio_data = await self.portfolio_tracker.get_portfolio_summary()

if portfolio_data:
# Update multi-profile manager with portfolio information
for profile_id in self.active_profiles.keys():
# Update profile metrics with portfolio data
if profile_id in self.multi_profile_manager.profile_metrics:
metrics = self.multi_profile_manager.profile_metrics[profile_id]
# Update metrics based on portfolio performance
# This is a simplified integration - expand based on actual portfolio data structure

except Exception as e:
self.logger.error(f"Error integrating portfolio data: {e}")

async def _integrate_risk_management(self):
"""Integrate risk management with multi-profile system."""
try:
# Get risk metrics from existing system
risk_metrics = await self.risk_manager.get_risk_metrics()

if risk_metrics:
# Apply risk adjustments to profiles
for profile_id, profile_config in self.active_profiles.items():
# Adjust profile risk parameters based on overall system risk
# This is a simplified integration - expand based on actual risk management structure
pass

except Exception as e:
self.logger.error(f"Error integrating risk management: {e}")

async def _update_routing_metrics(self):
"""Update routing performance metrics."""
try:
# Calculate routing success rate
if self.total_routed_trades > 0:
success_rate = self.successful_routes / self.total_routed_trades
self.logger.debug(f"📊 Routing success rate: {success_rate:.3f}")

# Log arbitration opportunities
if self.arbitration_opportunities > 0:
self.logger.info(f"💰 Total arbitration opportunities: {self.arbitration_opportunities}")

except Exception as e:
self.logger.error(f"Error updating routing metrics: {e}")

async def stop_trading(self):
"""Stop multi-profile trading."""
try:
self.running = False
self.logger.info("🛑 Stopping multi-profile trading...")

# Stop multi-profile manager
await self.multi_profile_manager.stop_trading()

self.logger.info("✅ Multi-profile trading stopped")

except Exception as e:
self.logger.error(f"Error stopping trading: {e}")

def get_profile_status(self) -> Dict[str, Any]:
"""Get status of all profiles."""
try:
# Get multi-profile manager status
multi_profile_status = self.multi_profile_manager.get_profile_status()

# Get strategy mapper status
strategy_status = self.strategy_mapper.get_system_status()

# Combine status information
status = {
'profile_router': {
'initialized': self.initialized,
'running': self.running,
'active_profiles': len(self.active_profiles),
'total_routed_trades': self.total_routed_trades,
'successful_routes': self.successful_routes,
'failed_routes': self.failed_routes,
'arbitration_opportunities': self.arbitration_opportunities,
'success_rate': self.successful_routes / self.total_routed_trades if self.total_routed_trades > 0 else 0.0
},
'multi_profile_manager': multi_profile_status,
'strategy_mapper': strategy_status,
'integration': {
'schwabot_components_available': SCHWABOT_COMPONENTS_AVAILABLE,
'live_trading_system_available': self.live_trading_system is not None,
'portfolio_tracker_available': self.portfolio_tracker is not None,
'risk_manager_available': self.risk_manager is not None
}
}

return status

except Exception as e:
self.logger.error(f"Error getting profile status: {e}")
return {}

def get_arbitration_history(self) -> List[Dict[str, Any]]:
"""Get cross-profile arbitration history."""
try:
return self.multi_profile_manager.get_arbitration_history()
except Exception as e:
self.logger.error(f"Error getting arbitration history: {e}")
return []

async def execute_unified_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
"""Execute trade through unified interface."""
try:
if not self.unified_interface_enabled:
raise RuntimeError("Unified interface not enabled")

# Route trade to appropriate profile
profile_id = self._select_profile_for_trade(trade_data)

if not profile_id:
return {'success': False, 'error': 'No suitable profile found'}

# Execute trade through selected profile
result = await self._execute_trade_on_profile(profile_id, trade_data)

return {
'success': True,
'profile_id': profile_id,
'result': result
}

except Exception as e:
self.logger.error(f"Error executing unified trade: {e}")
return {'success': False, 'error': str(e)}

def _select_profile_for_trade(self, trade_data: Dict[str, Any]) -> Optional[str]:
"""Select appropriate profile for trade execution."""
try:
# Simple profile selection logic - can be enhanced
symbol = trade_data.get('symbol', '')
side = trade_data.get('side', 'buy')
size = trade_data.get('size', 0)

# Select profile based on trade characteristics
for profile_id, profile_config in self.active_profiles.items():
trading_params = profile_config.get('trading_params', {})
trading_pairs = trading_params.get('trading_pairs', [])

# Check if symbol is supported by this profile
if any(symbol in pair for pair in trading_pairs):
# Check if profile has capacity
max_positions = trading_params.get('max_open_positions', 5)
current_positions = len(self.multi_profile_manager.profile_metrics.get(profile_id, {}).get('active_positions', []))

if current_positions < max_positions:
return profile_id

return None

except Exception as e:
self.logger.error(f"Error selecting profile for trade: {e}")
return None

async def _execute_trade_on_profile(self, profile_id: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
"""Execute trade on specific profile."""
try:
profile_api = self.multi_profile_manager.profiles.get(profile_id)
if not profile_api:
return {'success': False, 'error': 'Profile not available'}

# Prepare order parameters
order_params = {
'product_id': trade_data.get('symbol', '').replace('/', '-'),
'side': trade_data.get('side', 'buy'),
'type': trade_data.get('type', 'market'),
'size': str(trade_data.get('size', 0)),
'client_order_id': f"{profile_id}_{int(time.time())}"
}

# Place order
result = await profile_api.place_order(**order_params)

if result:
return {'success': True, 'order_id': result.get('id'), 'profile_id': profile_id}
else:
return {'success': False, 'error': 'Order placement failed'}

except Exception as e:
self.logger.error(f"Error executing trade on profile {profile_id}: {e}")
return {'success': False, 'error': str(e)}

def get_unified_interface_status(self) -> Dict[str, Any]:
"""Get unified interface status."""
try:
return {
'enabled': self.unified_interface_enabled,
'active_profiles': len(self.active_profiles),
'total_routed_trades': self.total_routed_trades,
'success_rate': self.successful_routes / self.total_routed_trades if self.total_routed_trades > 0 else 0.0,
'arbitration_opportunities': self.arbitration_opportunities
}
except Exception as e:
self.logger.error(f"Error getting unified interface status: {e}")
return {}
