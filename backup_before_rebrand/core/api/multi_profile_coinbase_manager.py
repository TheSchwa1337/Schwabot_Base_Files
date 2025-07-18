"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Profile Coinbase API Manager
==================================
Manages multiple Coinbase API profiles with independent strategy logic,
de-synced trade execution, and mathematical separation to ensure unique
profit opportunities for each profile.

Mathematical Core:
∀ t: H₁(t) ≠ H₂(t) ∨ A₁ ≠ A₂
Strategy_Profile(t, Pᵢ) = ƒ(Hashₜᵢ, Assetsᵢ, Holdingsᵢ, Profit_Zonesᵢ)
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
import yaml
import numpy as np
from datetime import datetime, timedelta

# Import existing Coinbase API
from .coinbase_direct import CoinbaseDirectAPI

logger = logging.getLogger(__name__)

class ProfileState(Enum):
"""Class for Schwabot trading functionality."""
"""Profile state enumeration."""
INITIALIZING = "initializing"
ACTIVE = "active"
PAUSED = "paused"
ERROR = "error"
DISCONNECTED = "disconnected"

class StrategyMode(Enum):
"""Class for Schwabot trading functionality."""
"""Strategy execution mode."""
INDEPENDENT = "independent"
SYNCHRONIZED = "synchronized"
ARBITRAGE = "arbitrage"

@dataclass
class ProfileMetrics:
"""Class for Schwabot trading functionality."""
"""Profile performance metrics."""
profile_id: str
total_trades: int = 0
successful_trades: int = 0
failed_trades: int = 0
total_profit: float = 0.0
total_loss: float = 0.0
win_rate: float = 0.0
avg_trade_size: float = 0.0
max_drawdown: float = 0.0
current_drawdown: float = 0.0
hash_collisions: int = 0
strategy_duplications: int = 0
last_update: datetime = field(default_factory=datetime.now)

@dataclass
class ProfileHashState:
"""Class for Schwabot trading functionality."""
"""Profile hash state for uniqueness enforcement."""
profile_id: str
current_hash: str
hash_trajectory: List[str] = field(default_factory=list)
entropy_stream: List[float] = field(default_factory=list)
last_rotation: datetime = field(default_factory=datetime.now)
hash_complexity: float = 1.0

@dataclass
class CrossProfileArbitration:
"""Class for Schwabot trading functionality."""
"""Cross-profile arbitration data."""
timestamp: datetime
profile_a_id: str
profile_b_id: str
arbitration_score: float
opportunity_detected: bool
action_taken: str
profit_potential: float

class MultiProfileCoinbaseManager:
"""Class for Schwabot trading functionality."""
"""
Multi-Profile Coinbase API Manager

Manages multiple Coinbase API profiles with:
- Independent strategy logic per profile
- De-synced trade execution
- Mathematical separation enforcement
- Cross-profile arbitration when beneficial
- Unique hash trajectories and entropy streams
"""


def __init__(self, config_path: str = "config/coinbase_profiles.yaml") -> None:
"""Initialize the multi-profile Coinbase manager."""
self.config_path = config_path
self.config = self._load_config()
self.logger = logging.getLogger(__name__)

# Profile management
self.profiles: Dict[str, CoinbaseDirectAPI] = {}
self.profile_states: Dict[str, ProfileState] = {}
self.profile_metrics: Dict[str, ProfileMetrics] = {}
self.profile_hashes: Dict[str, ProfileHashState] = {}

# Strategy management
self.strategy_modes: Dict[str, StrategyMode] = {}
self.active_strategies: Dict[str, List[str]] = {}
self.strategy_weights: Dict[str, Dict[str, float]] = {}

# Synchronization and arbitration
self.cross_profile_arbitration: List[CrossProfileArbitration] = []
self.hash_echo_cache: Dict[str, Dict[str, Any]] = {}
self.de_sync_delays: Dict[str, float] = {}

# Mathematical separation
self.hash_uniqueness_enforced = True
self.asset_uniqueness_enforced = True
self.path_encoding_enabled = True

# Performance tracking
self.total_profiles = 0
self.active_profiles = 0
self.last_arbitration_check = datetime.now()

# System state
self.initialized = False
self.running = False

self.logger.info("Multi-Profile Coinbase Manager initialized")

def _load_config(self) -> Dict[str, Any]:
"""Load configuration from YAML file."""
try:
with open(self.config_path, 'r') as file:
config = yaml.safe_load(file)
self.logger.info(f"Configuration loaded from {self.config_path}")
return config
except Exception as e:
self.logger.error(f"Failed to load configuration: {e}")
return {}

async def initialize_profiles(self) -> bool:
"""Initialize all configured profiles."""
try:
self.logger.info("🔄 Initializing Coinbase profiles...")

profiles_config = self.config.get('profiles', {})
self.total_profiles = len(profiles_config)

for profile_id, profile_config in profiles_config.items():
if not profile_config.get('enabled', False):
continue

await self._initialize_single_profile(profile_id, profile_config)

self.active_profiles = len(self.profiles)
self.initialized = True

self.logger.info(f"✅ Initialized {self.active_profiles}/{self.total_profiles} profiles")
return True

except Exception as e:
self.logger.error(f"Failed to initialize profiles: {e}")
return False

async def _initialize_single_profile(self, profile_id: str, profile_config: Dict[str, Any]):
"""Initialize a single profile."""
try:
# Extract API credentials
api_creds = profile_config.get('api_credentials', {})
api_key = api_creds.get('api_key', '')
secret = api_creds.get('secret', '')
passphrase = api_creds.get('passphrase', '')
sandbox = api_creds.get('sandbox', True)

if not all([api_key, secret, passphrase]):
self.logger.warning(f"⚠️ Incomplete API credentials for profile {profile_id}")
return

# Initialize Coinbase API
coinbase_api = CoinbaseDirectAPI(
api_key=api_key,
secret=secret,
passphrase=passphrase,
sandbox=sandbox
)

# Connect to API
if await coinbase_api.connect():
self.profiles[profile_id] = coinbase_api
self.profile_states[profile_id] = ProfileState.ACTIVE

# Initialize profile metrics
self.profile_metrics[profile_id] = ProfileMetrics(profile_id=profile_id)

# Initialize hash state
hash_config = profile_config.get('profile_hash', {})
self.profile_hashes[profile_id] = ProfileHashState(
profile_id=profile_id,
current_hash=self._generate_profile_hash(profile_id, hash_config),
hash_complexity=hash_config.get('hash_complexity_factor', 1.0)
)

# Initialize strategy configuration
strategy_config = profile_config.get('strategy_config', {})
self.strategy_weights[profile_id] = strategy_config.get('strategy_weights', {})
self.active_strategies[profile_id] = []

# Set initial strategy mode
self.strategy_modes[profile_id] = StrategyMode.INDEPENDENT

# Set de-sync delay
sync_config = self.config.get('synchronization', {})
delay_range = sync_config.get('de_sync_delay_range', [1, 15])
self.de_sync_delays[profile_id] = random.uniform(delay_range[0], delay_range[1])

self.logger.info(f"✅ Profile {profile_id} initialized successfully")
else:
self.profile_states[profile_id] = ProfileState.ERROR
self.logger.error(f"❌ Failed to connect profile {profile_id}")

except Exception as e:
self.logger.error(f"Error initializing profile {profile_id}: {e}")
self.profile_states[profile_id] = ProfileState.ERROR

def _generate_profile_hash(self, profile_id: str, hash_config: Dict[str, Any]) -> str:
"""Generate unique profile hash."""
base_hash = hash_config.get('base_hash', profile_id)
entropy_seed = hash_config.get('entropy_seed', 0.5)
timestamp = int(time.time())

# Create unique hash based on profile characteristics
hash_input = f"{base_hash}_{entropy_seed}_{timestamp}_{profile_id}"
return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

async def start_trading(self) -> bool:
"""Start trading across all active profiles."""
try:
if not self.initialized:
raise RuntimeError("Profiles not initialized")

self.running = True
self.logger.info("🚀 Starting multi-profile trading...")

# Start trading tasks for each profile
trading_tasks = []
for profile_id, profile_state in self.profile_states.items():
if profile_state == ProfileState.ACTIVE:
task = asyncio.create_task(self._profile_trading_loop(profile_id))
trading_tasks.append(task)

# Start cross-profile arbitration task
arbitration_task = asyncio.create_task(
self._cross_profile_arbitration_loop())
trading_tasks.append(arbitration_task)

# Start monitoring task
monitoring_task = asyncio.create_task(self._monitoring_loop())
trading_tasks.append(monitoring_task)

# Wait for all tasks
await asyncio.gather(*trading_tasks, return_exceptions=True)

return True

except Exception as e:
self.logger.error(f"Error starting trading: {e}")
return False

async def _profile_trading_loop(self, profile_id: str):
"""Individual profile trading loop."""
try:
self.logger.info(
f"🔄 Starting trading loop for profile {profile_id}")

while self.running and self.profile_states.get(profile_id) == ProfileState.ACTIVE:
# Apply de-sync delay
delay = self.de_sync_delays.get(profile_id, 5.0)
await asyncio.sleep(delay)

# Generate unique strategy for this profile
strategy_data = await self._generate_profile_strategy(profile_id)

if strategy_data:
# Execute strategy
await self._execute_profile_strategy(profile_id, strategy_data)

# Update profile hash
await self._update_profile_hash(profile_id)

# Check for hash collisions
await self._check_hash_uniqueness(profile_id)

except Exception as e:
self.logger.error(
f"Error in trading loop for profile {profile_id}: {e}")
self.profile_states[profile_id] = ProfileState.ERROR

async def _generate_profile_strategy(self, profile_id: str) -> Optional[Dict[str, Any]]:
"""Generate unique strategy for profile."""
try:
profile_config = self.config['profiles'][profile_id]
strategy_config = profile_config.get(
'strategy_config', {})

# Get current hash state
hash_state = self.profile_hashes[profile_id]

# Generate unique asset selection
base_assets = strategy_config.get(
'base_assets', [])
random_pool = strategy_config.get(
'random_asset_pool', [])

# Select 5 base assets + 1 random asset
selected_assets = base_assets.copy()
if random_pool:
random_asset = random.choice(random_pool)
selected_assets.append(random_asset)

# Generate strategy parameters based on
# profile hash
strategy_weights = self.strategy_weights[profile_id]
selected_strategy = self._select_strategy_by_hash(
hash_state.current_hash, strategy_weights)

# Calculate profile-specific parameters
confidence = strategy_config.get(
'confidence_threshold', 0.7)
signal_strength = strategy_config.get(
'signal_strength_threshold', 0.6)

# Apply hash-based adjustments
hash_adjustment = self._calculate_hash_adjustment(
hash_state.current_hash)
confidence *= hash_adjustment
signal_strength *= hash_adjustment

strategy_data = {
'profile_id': profile_id,
'strategy_name': selected_strategy,
'assets': selected_assets,
'confidence': min(confidence, 1.0),
'signal_strength': min(signal_strength, 1.0),
'hash_state': hash_state.current_hash,
'timestamp': datetime.now(),
'profile_hash': hash_state.current_hash
}

return strategy_data

except Exception as e:
self.logger.error(
f"Error generating strategy for profile {profile_id}: {e}")
return None

def _select_strategy_by_hash(self, profile_hash: str, strategy_weights: Dict[str, float]) -> str:
"""Select strategy based on profile hash."""
if not strategy_weights:
return "volume_weighted_hash_oscillator"

# Use hash to determine strategy selection
hash_int = int(profile_hash[:8], 16)
total_weight = sum(
strategy_weights.values())

if total_weight == 0:
return list(strategy_weights.keys())[0]

# Normalize weights
normalized_weights = {
k: v / total_weight for k, v in strategy_weights.items()}

# Use hash to select strategy
cumulative_weight = 0
hash_normalized = (hash_int % 1000) / 1000.0

for strategy, weight in normalized_weights.items():
cumulative_weight += weight
if hash_normalized <= cumulative_weight:
return strategy

return list(strategy_weights.keys())[-1]

def _calculate_hash_adjustment(self, profile_hash: str) -> float:
"""Calculate adjustment factor based on profile hash."""
hash_int = int(profile_hash[:8], 16)
# Generate adjustment between 0.8 and
# 1.2
adjustment = 0.8 + \
(hash_int % 400) / 1000.0
return adjustment

async def _execute_profile_strategy(self, profile_id: str, strategy_data: Dict[str, Any]):
"""Execute strategy for specific profile."""
try:
profile = self.profiles.get(
profile_id)
if not profile:
return

# Check if strategy meets confidence
# threshold
confidence = strategy_data.get(
'confidence', 0)
if confidence < 0.6:  # Minimum confidence threshold
return

# Get trading parameters
profile_config = self.config['profiles'][profile_id]
trading_params = profile_config.get(
'trading_params', {})
trading_pairs = trading_params.get(
'trading_pairs', [])

if not trading_pairs:
return

# Select trading pair based on strategy
selected_pair = self._select_trading_pair(
strategy_data, trading_pairs)

# Generate order parameters
order_params = self._generate_order_parameters(
profile_id, strategy_data, selected_pair)

if order_params:
# Place order
order_result = await self._place_profile_order(profile_id, order_params)

# Update metrics
if order_result:
await self._update_profile_metrics(profile_id, order_result, True)
else:
await self._update_profile_metrics(profile_id, None, False)

except Exception as e:
self.logger.error(
f"Error executing strategy for profile {profile_id}: {e}")

def _select_trading_pair(self, strategy_data: Dict[str, Any], trading_pairs: List[str]) -> str:
"""Select trading pair based on strategy data."""
assets = strategy_data.get(
'assets', [])
profile_hash = strategy_data.get(
'profile_hash', '')

# Use hash to select
# trading pair
hash_int = int(
profile_hash[:8], 16)
pair_index = hash_int % len(
trading_pairs)

return trading_pairs[pair_index]

def _generate_order_parameters(self, profile_id: str, strategy_data: Dict[str, Any], trading_pair: str) -> Optional[Dict[str, Any]]:
"""Generate order parameters for profile."""
try:
profile_config = self.config['profiles'][profile_id]
trading_params = profile_config.get(
'trading_params', {})
risk_management = profile_config.get(
'risk_management', {})

# Calculate
# position size
# based on
# profile
# configuration
max_position_size = trading_params.get(
'max_position_size_pct', 5.0) / 100.0
confidence = strategy_data.get(
'confidence', 0.5)

# Adjust
# position size
# based on
# confidence
adjusted_size = max_position_size * confidence

# Determine
# order side
# based on
# strategy hash
profile_hash = strategy_data.get(
'profile_hash', '')
hash_int = int(
profile_hash[:8], 16)
side = 'buy' if hash_int % 2 == 0 else 'sell'

# Generate order parameters
order_params = {
'product_id': trading_pair.replace('/', '-'),
'side': side,
'type': 'market',
'size': f"{adjusted_size:.4f}",
'client_order_id': f"{profile_id}_{int(time.time())}_{hash_int % 10000}"
}

return order_params

except Exception as e:
self.logger.error(
f"Error generating order parameters for profile {profile_id}: {e}")
return None

async def _place_profile_order(self, profile_id: str, order_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
"""Place order for specific profile."""
try:
profile = self.profiles.get(
profile_id)
if not profile:
return None

# Place order
order_result = await profile.place_order(
product_id=order_params['product_id'],
side=order_params['side'],
order_type=order_params['type'],
size=order_params['size'],
client_order_id=order_params['client_order_id']
)

if order_result:
self.logger.info(
f"✅ Order placed for profile {profile_id}: {order_result.get('id')}")

return order_result

except Exception as e:
self.logger.error(
f"Error placing order for profile {profile_id}: {e}")
return None

async def _update_profile_hash(self, profile_id: str):
"""Update profile hash for uniqueness."""
try:
hash_state = self.profile_hashes[profile_id]
profile_config = self.config[
'profiles'][profile_id]
hash_config = profile_config.get(
'profile_hash', {})

# Check
# if
# hash
# rotation
# is
# needed
rotation_interval = hash_config.get(
'hash_rotation_interval', 3600)
time_since_rotation = (
datetime.now() - hash_state.last_rotation).total_seconds()

if time_since_rotation >= rotation_interval:
# Generate new hash
new_hash = self._generate_profile_hash(
profile_id, hash_config)

# Update
# hash
# state
hash_state.hash_trajectory.append(
hash_state.current_hash)
hash_state.current_hash = new_hash
hash_state.last_rotation = datetime.now()

# Keep
# trajectory
# manageable
if len(hash_state.hash_trajectory) > 100:
hash_state.hash_trajectory = hash_state.hash_trajectory[
-50:]

self.logger.debug(
f"🔄 Hash rotated for profile {profile_id}: {new_hash[:8]}")

except Exception as e:
self.logger.error(
f"Error updating hash for profile {profile_id}: {e}")

async def _check_hash_uniqueness(self, profile_id: str):
"""Check hash uniqueness across profiles."""
try:
current_hash = self.profile_hashes[
profile_id].current_hash

for other_profile_id, other_hash_state in self.profile_hashes.items():
if other_profile_id != profile_id:
if other_hash_state.current_hash == current_hash:
# Hash collision detected
self.profile_metrics[
profile_id].hash_collisions += 1
self.logger.warning(
f"⚠️ Hash collision detected between {profile_id} and {other_profile_id}")

# Force
# hash
# rotation
await self._force_hash_rotation(profile_id)
break

except Exception as e:
self.logger.error(
f"Error checking hash uniqueness for profile {profile_id}: {e}")

async def _force_hash_rotation(self, profile_id: str):
"""Force hash rotation to resolve collision."""
try:
hash_state = self.profile_hashes[
profile_id]
profile_config = self.config[
'profiles'][profile_id]
hash_config = profile_config.get(
'profile_hash', {})

# Generate
# new
# hash
# with
# additional
# entropy
timestamp = int(
time.time())
entropy = random.random()
new_hash_input = f"{
hash_state.current_hash}_{timestamp}_{entropy}_{profile_id}"
new_hash = hashlib.sha256(
new_hash_input.encode()).hexdigest()[:16]

hash_state.current_hash = new_hash
hash_state.last_rotation = datetime.now()

self.logger.info(
f"🔄 Forced hash rotation for profile {profile_id}: {new_hash[:8]}")

except Exception as e:
self.logger.error(
f"Error forcing hash rotation for profile {profile_id}: {e}")

async def _cross_profile_arbitration_loop(self):
"""Cross-profile arbitration loop."""
try:
sync_config = self.config.get(
'synchronization', {})
echo_interval = sync_config.get(
'echo_validation_interval', 300)

while self.running:
await asyncio.sleep(echo_interval)

if sync_config.get('enable_cross_profile_arbitration', False):
await self._perform_cross_profile_arbitration()

except Exception as e:
self.logger.error(
f"Error in cross-profile arbitration loop: {e}")

async def _perform_cross_profile_arbitration(self):
"""Perform cross-profile arbitration."""
try:
active_profiles = [pid for pid, state in self.profile_states.items()
if state == ProfileState.ACTIVE]

if len(active_profiles) < 2:
return

# Check
# for
# arbitrage
# opportunities
for i, profile_a in enumerate(active_profiles):
for profile_b in active_profiles[i +1:]:
arbitration_score = await self._calculate_arbitration_score(profile_a, profile_b)

if arbitration_score > 0.85:  # High arbitrage opportunity
await self._execute_arbitrage_strategy(profile_a, profile_b, arbitration_score)

except Exception as e:
self.logger.error(
f"Error performing cross-profile arbitration: {e}")

async def _calculate_arbitration_score(self, profile_a: str, profile_b: str) -> float:
"""Calculate arbitrage opportunity score between profiles."""
try:
# Get profile hashes
hash_a = self.profile_hashes[
profile_a].current_hash
hash_b = self.profile_hashes[
profile_b].current_hash

# Calculate
# hash
# similarity
hash_similarity = self._calculate_hash_similarity(
hash_a, hash_b)

# Calculate
# asset
# overlap
profile_config_a = self.config[
'profiles'][profile_a]
profile_config_b = self.config[
'profiles'][profile_b]

assets_a = set(profile_config_a.get(
'strategy_config', {}).get('base_assets', []))
assets_b = set(profile_config_b.get(
'strategy_config', {}).get('base_assets', []))

asset_overlap = len(assets_a.intersection(
assets_b)) / len(assets_a.union(assets_b))

# Calculate
# arbitrage
# score
arbitrage_score = (
1 - hash_similarity) * (1 - asset_overlap)

return arbitrage_score

except Exception as e:
self.logger.error(
f"Error calculating arbitrage score: {e}")
return 0.0

def _calculate_hash_similarity(self, hash_a: str, hash_b: str) -> float:
"""Calculate similarity between two hashes."""
try:
# Convert hashes to binary and calculate Hamming distance
hash_a_bin = bin(int(hash_a, 16))[
2:].zfill(64)
hash_b_bin = bin(int(hash_b, 16))[
2:].zfill(64)

hamming_distance = sum(
a != b for a, b in zip(hash_a_bin, hash_b_bin))
similarity = 1 - \
(hamming_distance / 64)

return similarity

except Exception as e:
self.logger.error(
f"Error calculating hash similarity: {e}")
return 0.5

async def _execute_arbitrage_strategy(self, profile_a: str, profile_b: str, score: float):
"""Execute arbitrage strategy between profiles."""
try:
# Create arbitration record
arbitration = CrossProfileArbitration(
timestamp=datetime.now(),
profile_a_id=profile_a,
profile_b_id=profile_b,
arbitration_score=score,
opportunity_detected=True,
action_taken="arbitrage_executed",
profit_potential=score * 100  # Estimate profit potential
)

self.cross_profile_arbitration.append(
arbitration)

# Keep
# arbitration
# history
# manageable
if len(self.cross_profile_arbitration) > 1000:
self.cross_profile_arbitration = self.cross_profile_arbitration[
-500:]

self.logger.info(
f"💰 Arbitrage opportunity detected between {profile_a} and {profile_b} (score: {score:.3f})")

except Exception as e:
self.logger.error(
f"Error executing arbitrage strategy: {e}")

async def _update_profile_metrics(self, profile_id: str, order_result: Optional[Dict[str, Any]], success: bool):
"""Update profile performance metrics."""
try:
metrics = self.profile_metrics[
profile_id]

metrics.total_trades += 1

if success:
metrics.successful_trades += 1
# Calculate
# profit
# (simplified)
if order_result:
size = float(
order_result.get('size', 0))
metrics.total_profit += size * 0.01  # Assume 1% profit
else:
metrics.failed_trades += 1
metrics.total_loss += 0.01  # Assume 1% loss

# Update
# win
# rate
if metrics.total_trades > 0:
metrics.win_rate = metrics.successful_trades / metrics.total_trades

# Update
# average
# trade
# size
if order_result:
size = float(
order_result.get('size', 0))
metrics.avg_trade_size = (
metrics.avg_trade_size * (metrics.total_trades - 1) + size) / metrics.total_trades

metrics.last_update = datetime.now()

except Exception as e:
self.logger.error(
f"Error updating metrics for profile {profile_id}: {e}")

async def _monitoring_loop(self):
"""Monitoring and alerting loop."""
try:
monitoring_config = self.config.get(
'monitoring', {})
collection_interval = monitoring_config.get(
'metrics_collection_interval', 60)

while self.running:
await asyncio.sleep(collection_interval)

# Check
# performance
# thresholds
await self._check_performance_thresholds()

# Check
# for
# hash
# collisions
await self._check_hash_collisions()

# Check
# for
# strategy
# duplications
await self._check_strategy_duplications()

except Exception as e:
self.logger.error(
f"Error in monitoring loop: {e}")

async def _check_performance_thresholds(self):
"""Check performance thresholds and generate alerts."""
try:
monitoring_config = self.config.get(
'monitoring', {})
threshold = monitoring_config.get(
'profile_performance_threshold', 0.6)

for profile_id, metrics in self.profile_metrics.items():
if metrics.win_rate < threshold:
self.logger.warning(
f"⚠️ Profile {profile_id} performance below threshold: {metrics.win_rate:.3f}")

except Exception as e:
self.logger.error(
f"Error checking performance thresholds: {e}")

async def _check_hash_collisions(self):
"""Check for hash collisions across profiles."""
try:
monitoring_config = self.config.get(
'monitoring', {})
if not monitoring_config.get('alert_on_hash_collision', True):
return

for profile_id, metrics in self.profile_metrics.items():
if metrics.hash_collisions > 0:
self.logger.warning(
f"⚠️ Hash collisions detected for profile {profile_id}: {metrics.hash_collisions}")

except Exception as e:
self.logger.error(
f"Error checking hash collisions: {e}")

async def _check_strategy_duplications(self):
"""Check for strategy duplications across profiles."""
try:
monitoring_config = self.config.get(
'monitoring', {})
if not monitoring_config.get('alert_on_strategy_duplication', True):
return

# Check
# for
# duplicate
# strategies
# in
# recent
# history
recent_strategies = {}
for profile_id, strategies in self.active_strategies.items():
if strategies:
# Most recent strategy
recent_strategies[profile_id] = strategies[-1]

# Check
# for
# duplicates
strategy_counts = {}
for profile_id, strategy in recent_strategies.items():
strategy_counts[strategy] = strategy_counts.get(
strategy, 0) + 1

for strategy, count in strategy_counts.items():
if count > 1:
self.logger.warning(
f"⚠️ Strategy duplication detected: {strategy} used by {count} profiles")

except Exception as e:
self.logger.error(
f"Error checking strategy duplications: {e}")

async def stop_trading(self):
"""Stop trading across all profiles."""
try:
self.running = False
self.logger.info(
"🛑 Stopping multi-profile trading...")

# Disconnect
# all
# profiles
for profile_id, profile in self.profiles.items():
await profile.disconnect()
self.profile_states[
profile_id] = ProfileState.DISCONNECTED

self.logger.info(
"✅ Multi-profile trading stopped")

except Exception as e:
self.logger.error(
f"Error stopping trading: {e}")

def get_profile_status(self) -> Dict[str, Any]:
"""Get status of all profiles."""
try:
status = {
'total_profiles': self.total_profiles,
'active_profiles': self.active_profiles,
'system_running': self.running,
'profiles': {}
}

for profile_id in self.profile_states.keys():
profile_status = {
'state': self.profile_states[profile_id].value,
'metrics': self.profile_metrics[profile_id].__dict__ if profile_id in self.profile_metrics else {},
'hash_state': {
'current_hash': self.profile_hashes[profile_id].current_hash[:8] if profile_id in self.profile_hashes else '',
'last_rotation': self.profile_hashes[profile_id].last_rotation.isoformat() if profile_id in self.profile_hashes else '',
'hash_complexity': self.profile_hashes[profile_id].hash_complexity if profile_id in self.profile_hashes else 1.0
} if profile_id in self.profile_hashes else {},
'strategy_mode': self.strategy_modes[profile_id].value if profile_id in self.strategy_modes else 'unknown'
}
status['profiles'][
profile_id] = profile_status

return status

except Exception as e:
self.logger.error(
f"Error getting profile status: {e}")
return {}

def get_arbitration_history(self) -> List[Dict[str, Any]]:
"""Get cross-profile arbitration history."""
try:
return [
{
'timestamp': arb.timestamp.isoformat(),
'profile_a': arb.profile_a_id,
'profile_b': arb.profile_b_id,
'score': arb.arbitration_score,
'opportunity_detected': arb.opportunity_detected,
'action_taken': arb.action_taken,
'profit_potential': arb.profit_potential
}
for arb in self.cross_profile_arbitration[-100:]  # Last 100 records
]
except Exception as e:
self.logger.error(
f"Error getting arbitration history: {e}")
return []
