"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDC Trading Engine 🚀

Real BTC/USDC trading engine implementing your mathematical framework:
• Strategy matrices → profit matrices → tensor calculations
• Entry/exit functions based on mathematical triggers
• Ghost basket internal state management
• YAML configuration-driven mathematical implementations
• Thermal-aware and multi-bit processing integration

Features:
- Real mathematical implementations from your YAML configs
- BTC/USDC specific trading logic (not generic arbitrage)
- Strategy matrix to profit matrix pipeline
- Tensor calculations for entry/exit decisions
- Internal state management (ghost baskets)
- Thermal and multi-bit processing integration
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

try:
import cupy as cp
import numpy as np
USING_CUDA = True
xp = cp
_backend = 'cupy (GPU)'
except ImportError:
try:
import numpy as np
USING_CUDA = False
xp = np
_backend = 'numpy (CPU)'
except ImportError:
xp = None
_backend = 'none'

logger = logging.getLogger(__name__)
if xp is None:
logger.warning("❌ NumPy not available for tensor operations")
else:
logger.info(f"⚡ BTCTradingEngine using {_backend} for tensor operations")

class TradingMode(Enum):
"""Class for Schwabot trading functionality."""
"""Trading mode based on thermal state."""
OPTIMAL_AGGRESSIVE = "optimal_aggressive"
BALANCED_CONSISTENT = "balanced_consistent"
EFFICIENT_CONSERVATIVE = "efficient_conservative"
THROTTLE_SAFETY = "throttle_safety"
CRITICAL_HALT = "critical_halt"


class BitLevel(Enum):
"""Class for Schwabot trading functionality."""
"""Bit processing levels."""
BIT_4 = 4
BIT_8 = 8
BIT_16 = 16
BIT_32 = 32
BIT_42 = 42  # Phaser level
BIT_64 = 64  # Deep analysis


@dataclass
class BTCPriceData:
"""Class for Schwabot trading functionality."""
"""BTC price data with mathematical analysis."""
timestamp: int
price: float
volume: float
hash_value: str
bit_phase: int
tensor_score: float = 0.0
entropy: float = 0.0
thermal_state: float = 65.0  # Default temperature


@dataclass
class StrategyMatrix:
"""Class for Schwabot trading functionality."""
"""Strategy matrix for BTC/USDC trading."""
matrix_id: str
entry_conditions: Dict[str, Any]
exit_conditions: Dict[str, Any]
profit_targets: Dict[str, float]
stop_losses: Dict[str, float]
tensor_weights: xp.ndarray
confidence_score: float
bit_level: BitLevel
thermal_mode: TradingMode


@dataclass
class ProfitMatrix:
"""Class for Schwabot trading functionality."""
"""Profit matrix with tensor calculations."""
matrix_id: str
entry_price: float
current_price: float
position_size: float
unrealized_pnl: float
tensor_profit_score: float
profit_target: float
stop_loss: float
confidence: float
bit_phase: int
thermal_state: float


@dataclass
class GhostBasket:
"""Class for Schwabot trading functionality."""
"""Internal state management (ghost basket)."""
basket_id: str
strategy_matrices: List[StrategyMatrix]
profit_matrices: List[ProfitMatrix]
total_value: float
total_pnl: float
risk_metrics: Dict[str, float]
last_update: int


@dataclass
class TradingSignal:
"""Class for Schwabot trading functionality."""
"""Trading signal with mathematical validation."""
signal_type: str  # 'buy', 'sell', 'hold'
symbol: str = "BTC/USDC"
price: float = 0.0
amount: float = 0.0
confidence: float = 0.0
tensor_score: float = 0.0
bit_phase: int = 0
thermal_mode: TradingMode = TradingMode.BALANCED_CONSISTENT
strategy_matrix_id: str = ""
metadata: Dict[str, Any] = field(default_factory=dict)


class BTCTradingEngine:
"""Class for Schwabot trading functionality."""
"""
BTC/USDC Trading Engine implementing your mathematical framework.
Handles strategy matrices, profit matrices, tensor calculations, and ghost baskets.
"""
def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
self.config = config or self._load_default_config()
self.price_history: List[BTCPriceData] = []
self.strategy_matrices: Dict[str, StrategyMatrix] = {}
self.profit_matrices: Dict[str, ProfitMatrix] = {}
self.ghost_baskets: Dict[str, GhostBasket] = {}
self.trading_signals: List[TradingSignal] = []

# Mathematical parameters from YAML configs
self.thermal_thresholds = self.config.get('thermal_thresholds', {})
self.bit_level_configs = self.config.get('bit_level_configs', {})
self.profit_targets = self.config.get('profit_targets', {})
self.risk_limits = self.config.get('risk_limits', {})

# Initialize mathematical framework
self._initialize_mathematical_framework()

logger.info("✅ BTC/USDC Trading Engine initialized")

def _load_default_config(self) -> Dict[str, Any]:
"""Load default configuration from YAML files."""
return {
'thermal_thresholds': {
'optimal_performance': 65.0,
'balanced_processing': 75.0,
'thermal_efficient': 85.0,
'emergency_throttle': 90.0,
'critical_protection': 95.0
},
'bit_level_configs': {
4: {'signal_strength': 'noise', 'confidence_threshold': 0.9, 'position_multiplier': 0.3},
8: {'signal_strength': 'low', 'confidence_threshold': 0.8, 'position_multiplier': 0.5},
16: {'signal_strength': 'medium', 'confidence_threshold': 0.75, 'position_multiplier': 1.0},
32: {'signal_strength': 'high', 'confidence_threshold': 0.7, 'position_multiplier': 1.2},
42: {'signal_strength': 'critical', 'confidence_threshold': 0.65, 'position_multiplier': 1.5},
64: {'signal_strength': 'critical', 'confidence_threshold': 0.6, 'position_multiplier': 1.8}
},
'profit_targets': {
'base_profit_target_bp': 10,  # 0.1%
'base_stop_loss_bp': 5,       # 0.05%
'slippage_tolerance_bp': 2    # 0.02%
},
'risk_limits': {
'max_position_size_btc': 0.01,
'max_positions': 10,
'max_correlation': 0.7
}
}

def _initialize_mathematical_framework(self) -> None:
"""Initialize the mathematical framework components."""
try:
# Initialize tensor operations
if xp is None:
raise ValueError("Tensor operations not available")

# Initialize mathematical constants from config
self.psi_infinity = 1.618033988749  # Golden ratio
self.drift_coefficient = 0.1
self.entropy_threshold = 2.5
self.fit_threshold = 0.85

logger.info("✅ Mathematical framework initialized")

except Exception as e:
logger.error(f"❌ Failed to initialize mathematical framework: {e}")
raise

def process_btc_price(self, price: float, volume: float, thermal_state: float = 65.0) -> BTCPriceData:
"""Process BTC price data with mathematical analysis."""
try:
timestamp = int(time.time() * 1000)

# Generate hash from price and volume
price_str = f"{price:.2f}_{volume:.2f}_{timestamp}"
hash_value = hashlib.sha256(price_str.encode()).hexdigest()

# Calculate bit phase from hash
bit_phase = self._resolve_bit_phase(hash_value, "16bit")

# Calculate tensor score
tensor_score = self._calculate_tensor_score(price, volume, bit_phase)

# Calculate entropy
entropy = self._calculate_entropy(price, volume)

price_data = BTCPriceData(
timestamp=timestamp,
price=price,
volume=volume,
hash_value=hash_value,
bit_phase=bit_phase,
tensor_score=tensor_score,
entropy=entropy,
thermal_state=thermal_state
)

self.price_history.append(price_data)

# Keep only recent history
if len(self.price_history) > 1000:
self.price_history.pop(0)

return price_data

except Exception as e:
logger.error(f"❌ Failed to process BTC price: {e}")
raise

def _resolve_bit_phase(self, hash_str: str, mode: str = "16bit") -> int:
"""Resolve bit phase from hash string with SHA-256 decoding."""
try:
if mode == "4bit":
return int(hash_str[:1], 16) % 16
elif mode == "8bit":
return int(hash_str[:2], 16) % 256
elif mode == "16bit":
return int(hash_str[:4], 16) % 65536
elif mode == "32bit":
return int(hash_str[:8], 16) % 4294967296
elif mode == "42bit":
return int(hash_str[:11], 16) % (2**42)
elif mode == "64bit":
return int(hash_str[:16], 16) % (2**64)
else:
return int(hash_str[:4], 16) % 65536  # Default to 16bit

except Exception as e:
logger.warning(f"⚠️ Failed to resolve bit phase: {e}")
return 0

def _calculate_tensor_score(self, price: float, volume: float, phase: int) -> float:
"""Calculate tensor score for profit allocation."""
try:
if len(self.price_history) < 2:
return 0.0

# Get recent price data
recent_prices = [p.price for p in self.price_history[-10:]]
if len(recent_prices) < 2:
return 0.0

# Calculate price change
current_price = recent_prices[-1]
entry_price = recent_prices[0]

if entry_price == 0:
return 0.0

# Tensor score formula: ((current - entry) / entry) * (phase + 1)
price_change_ratio = (current_price - entry_price) / entry_price
tensor_score = price_change_ratio * (phase + 1)

return float(xp.clip(tensor_score, -10.0, 10.0))

except Exception as e:
logger.warning(f"⚠️ Failed to calculate tensor score: {e}")
return 0.0

def _calculate_entropy(self, price: float, volume: float) -> float:
"""Calculate entropy of price and volume data."""
try:
if len(self.price_history) < 10:
return 0.0

# Get recent price changes
recent_prices = [p.price for p in self.price_history[-10:]]
price_changes = xp.diff(recent_prices)

# Normalize price changes
if xp.std(price_changes) > 0:
normalized_changes = (price_changes - xp.mean(price_changes)) / xp.std(price_changes)
else:
normalized_changes = price_changes

# Calculate Shannon entropy
abs_changes = xp.abs(normalized_changes)
p = abs_changes / xp.sum(abs_changes) if xp.sum(abs_changes) > 0 else abs_changes
p = p[p > 0]

entropy = -xp.sum(p * xp.log(p)) if len(p) > 0 else 0.0
return float(entropy)

except Exception as e:
logger.warning(f"⚠️ Failed to calculate entropy: {e}")
return 0.0

def create_strategy_matrix(self, matrix_id: str, bit_level: BitLevel, -> None
thermal_mode: TradingMode) -> StrategyMatrix:
"""Create a strategy matrix for BTC/USDC trading."""
try:
# Get bit level configuration
bit_config = self.bit_level_configs.get(bit_level.value, {})
confidence_threshold = bit_config.get('confidence_threshold', 0.75)
position_multiplier = bit_config.get('position_multiplier', 1.0)

# Calculate tensor weights based on bit level
tensor_weights = self._calculate_tensor_weights(bit_level)

# Create entry/exit conditions based on thermal mode
entry_conditions = self._create_entry_conditions(thermal_mode, bit_level)
exit_conditions = self._create_exit_conditions(thermal_mode, bit_level)

# Calculate profit targets and stop losses
profit_targets = self._calculate_profit_targets(thermal_mode, bit_level)
stop_losses = self._calculate_stop_losses(thermal_mode, bit_level)

strategy_matrix = StrategyMatrix(
matrix_id=matrix_id,
entry_conditions=entry_conditions,
exit_conditions=exit_conditions,
profit_targets=profit_targets,
stop_losses=stop_losses,
tensor_weights=tensor_weights,
confidence_score=confidence_threshold,
bit_level=bit_level,
thermal_mode=thermal_mode
)

self.strategy_matrices[matrix_id] = strategy_matrix
logger.info(f"✅ Created strategy matrix: {matrix_id}")

return strategy_matrix

except Exception as e:
logger.error(f"❌ Failed to create strategy matrix: {e}")
raise

def _calculate_tensor_weights(self, bit_level: BitLevel) -> xp.ndarray:
"""Calculate tensor weights based on bit level."""
try:
# Create weights based on bit level complexity
if bit_level == BitLevel.BIT_4:
weights = xp.array([0.25, 0.25, 0.25, 0.25])
elif bit_level == BitLevel.BIT_8:
weights = xp.array([0.125] * 8)
elif bit_level == BitLevel.BIT_16:
weights = xp.array([0.0625] * 16)
elif bit_level == BitLevel.BIT_32:
weights = xp.array([0.03125] * 32)
elif bit_level == BitLevel.BIT_42:
weights = xp.array([0.0238] * 42)  # 1/42
elif bit_level == BitLevel.BIT_64:
weights = xp.array([0.015625] * 64)  # 1/64
else:
weights = xp.array([0.0625] * 16)  # Default to 16-bit

return weights

except Exception as e:
logger.warning(f"⚠️ Failed to calculate tensor weights: {e}")
return xp.array([0.0625] * 16)

def _create_entry_conditions(self, thermal_mode: TradingMode, bit_level: BitLevel) -> Dict[str, Any]:
"""Create entry conditions based on thermal mode and bit level."""
base_conditions = {
'min_confidence': 0.75,
'min_tensor_score': 0.5,
'max_entropy': 2.5,
'min_volume': 1000.0
}

# Adjust based on thermal mode
if thermal_mode == TradingMode.OPTIMAL_AGGRESSIVE:
base_conditions['min_confidence'] *= 0.8
base_conditions['min_tensor_score'] *= 0.8
base_conditions['max_entropy'] *= 1.2
elif thermal_mode == TradingMode.EFFICIENT_CONSERVATIVE:
base_conditions['min_confidence'] *= 1.2
base_conditions['min_tensor_score'] *= 1.2
base_conditions['max_entropy'] *= 0.8
elif thermal_mode == TradingMode.THROTTLE_SAFETY:
base_conditions['min_confidence'] *= 1.5
base_conditions['min_tensor_score'] *= 1.5
base_conditions['max_entropy'] *= 0.6
elif thermal_mode == TradingMode.CRITICAL_HALT:
base_conditions['min_confidence'] = 1.0  # No trading

# Adjust based on bit level
bit_config = self.bit_level_configs.get(bit_level.value, {})
confidence_threshold = bit_config.get('confidence_threshold', 0.75)
base_conditions['min_confidence'] = max(base_conditions['min_confidence'], confidence_threshold)

return base_conditions

def _create_exit_conditions(self, thermal_mode: TradingMode, bit_level: BitLevel) -> Dict[str, Any]:
"""Create exit conditions based on thermal mode and bit level."""
base_conditions = {
'max_loss_ratio': 0.05,  # 5% max loss
'min_profit_ratio': 0.02,  # 2% min profit
'max_hold_time': 3600,  # 1 hour max hold
'stop_loss_trigger': 0.03  # 3% stop loss trigger
}

# Adjust based on thermal mode
if thermal_mode == TradingMode.OPTIMAL_AGGRESSIVE:
base_conditions['max_hold_time'] *= 1.5
base_conditions['min_profit_ratio'] *= 0.8
elif thermal_mode == TradingMode.EFFICIENT_CONSERVATIVE:
base_conditions['max_hold_time'] *= 0.8
base_conditions['min_profit_ratio'] *= 1.2
elif thermal_mode == TradingMode.THROTTLE_SAFETY:
base_conditions['max_hold_time'] *= 0.5
base_conditions['min_profit_ratio'] *= 1.5
elif thermal_mode == TradingMode.CRITICAL_HALT:
base_conditions['max_hold_time'] = 300  # 5 minutes max

return base_conditions

def _calculate_profit_targets(self, thermal_mode: TradingMode, bit_level: BitLevel) -> Dict[str, float]:
"""Calculate profit targets based on thermal mode and bit level."""
base_profit_bp = self.profit_targets.get('base_profit_target_bp', 10)

# Adjust based on thermal mode
if thermal_mode == TradingMode.OPTIMAL_AGGRESSIVE:
profit_multiplier = 1.2
elif thermal_mode == TradingMode.BALANCED_CONSISTENT:
profit_multiplier = 1.0
elif thermal_mode == TradingMode.EFFICIENT_CONSERVATIVE:
profit_multiplier = 0.8
elif thermal_mode == TradingMode.THROTTLE_SAFETY:
profit_multiplier = 0.6
elif thermal_mode == TradingMode.CRITICAL_HALT:
profit_multiplier = 0.0
else:
profit_multiplier = 1.0

# Adjust based on bit level
bit_config = self.bit_level_configs.get(bit_level.value, {})
position_multiplier = bit_config.get('position_multiplier', 1.0)

profit_target_bp = base_profit_bp * profit_multiplier * position_multiplier

return {
'profit_target_bp': profit_target_bp,
'profit_target_ratio': profit_target_bp / 10000.0
}

def _calculate_stop_losses(self, thermal_mode: TradingMode, bit_level: BitLevel) -> Dict[str, float]:
"""Calculate stop losses based on thermal mode and bit level."""
base_stop_loss_bp = self.profit_targets.get('base_stop_loss_bp', 5)

# Adjust based on thermal mode
if thermal_mode == TradingMode.OPTIMAL_AGGRESSIVE:
stop_loss_multiplier = 1.2
elif thermal_mode == TradingMode.BALANCED_CONSISTENT:
stop_loss_multiplier = 1.0
elif thermal_mode == TradingMode.EFFICIENT_CONSERVATIVE:
stop_loss_multiplier = 0.8
elif thermal_mode == TradingMode.THROTTLE_SAFETY:
stop_loss_multiplier = 0.6
elif thermal_mode == TradingMode.CRITICAL_HALT:
stop_loss_multiplier = 0.3
else:
stop_loss_multiplier = 1.0

stop_loss_bp = base_stop_loss_bp * stop_loss_multiplier

return {
'stop_loss_bp': stop_loss_bp,
'stop_loss_ratio': stop_loss_bp / 10000.0
}

def generate_trading_signal(self, price_data: BTCPriceData) -> Optional[TradingSignal]:
"""Generate trading signal based on mathematical analysis."""
try:
if len(self.price_history) < 10:
return None

# Determine thermal mode
thermal_mode = self._determine_thermal_mode(price_data.thermal_state)

# Determine bit level
bit_level = self._determine_bit_level(price_data.bit_phase)

# Calculate signal confidence
confidence = self._calculate_signal_confidence(price_data, thermal_mode, bit_level)

# Check if signal meets thresholds
if confidence < 0.75:  # Minimum confidence threshold
return None

# Determine signal type
signal_type = self._determine_signal_type(price_data, confidence)

if signal_type == 'hold':
return None

# Calculate position size
amount = self._calculate_position_size(price_data, thermal_mode, bit_level)

# Create trading signal
signal = TradingSignal(
signal_type=signal_type,
price=price_data.price,
amount=amount,
confidence=confidence,
tensor_score=price_data.tensor_score,
bit_phase=price_data.bit_phase,
thermal_mode=thermal_mode,
metadata={
'hash_value': price_data.hash_value,
'entropy': price_data.entropy,
'volume': price_data.volume
}
)

self.trading_signals.append(signal)

logger.info(f"📊 Generated {signal_type} signal: confidence={confidence:.3f}, "
f"tensor_score={price_data.tensor_score:.3f}, bit_phase={price_data.bit_phase}")

return signal

except Exception as e:
logger.error(f"❌ Failed to generate trading signal: {e}")
return None

def _determine_thermal_mode(self, thermal_state: float) -> TradingMode:
"""Determine trading mode based on thermal state."""
thresholds = self.thermal_thresholds

if thermal_state <= thresholds.get('optimal_performance', 65.0):
return TradingMode.OPTIMAL_AGGRESSIVE
elif thermal_state <= thresholds.get('balanced_processing', 75.0):
return TradingMode.BALANCED_CONSISTENT
elif thermal_state <= thresholds.get('thermal_efficient', 85.0):
return TradingMode.EFFICIENT_CONSERVATIVE
elif thermal_state <= thresholds.get('emergency_throttle', 90.0):
return TradingMode.THROTTLE_SAFETY
else:
return TradingMode.CRITICAL_HALT

def _determine_bit_level(self, bit_phase: int) -> BitLevel:
"""Determine bit level based on bit phase."""
if bit_phase < 16:
return BitLevel.BIT_4
elif bit_phase < 256:
return BitLevel.BIT_8
elif bit_phase < 65536:
return BitLevel.BIT_16
elif bit_phase < 4294967296:
return BitLevel.BIT_32
elif bit_phase < 2**42:
return BitLevel.BIT_42
else:
return BitLevel.BIT_64

def _calculate_signal_confidence(self, price_data: BTCPriceData, -> None
thermal_mode: TradingMode, bit_level: BitLevel) -> float:
"""Calculate signal confidence based on multiple factors."""
try:
# Base confidence from tensor score
tensor_confidence = min(1.0, abs(price_data.tensor_score) / 5.0)

# Entropy confidence (lower entropy = higher confidence)
entropy_confidence = max(0.0, 1.0 - (price_data.entropy / 5.0))

# Bit level confidence
bit_config = self.bit_level_configs.get(bit_level.value, {})
bit_confidence = bit_config.get('confidence_threshold', 0.75)

# Thermal mode confidence
thermal_confidence = {
TradingMode.OPTIMAL_AGGRESSIVE: 1.0,
TradingMode.BALANCED_CONSISTENT: 0.9,
TradingMode.EFFICIENT_CONSERVATIVE: 0.8,
TradingMode.THROTTLE_SAFETY: 0.6,
TradingMode.CRITICAL_HALT: 0.0
}.get(thermal_mode, 0.5)

# Weighted average
confidence = (
tensor_confidence * 0.4 +
entropy_confidence * 0.3 +
bit_confidence * 0.2 +
thermal_confidence * 0.1
)

return float(xp.clip(confidence, 0.0, 1.0))

except Exception as e:
logger.warning(f"⚠️ Failed to calculate signal confidence: {e}")
return 0.5

def _determine_signal_type(self, price_data: BTCPriceData, confidence: float) -> str:
"""Determine signal type (buy/sell/hold) based on analysis."""
try:
# Use tensor score to determine direction
if price_data.tensor_score > 1.0 and confidence > 0.8:
return 'buy'
elif price_data.tensor_score < -1.0 and confidence > 0.8:
return 'sell'
else:
return 'hold'

except Exception as e:
logger.warning(f"⚠️ Failed to determine signal type: {e}")
return 'hold'

def _calculate_position_size(self, price_data: BTCPriceData, -> None
thermal_mode: TradingMode, bit_level: BitLevel) -> float:
"""Calculate position size based on thermal mode and bit level."""
try:
base_position = self.risk_limits.get('max_position_size_btc', 0.01)

# Get bit level multiplier
bit_config = self.bit_level_configs.get(bit_level.value, {})
position_multiplier = bit_config.get('position_multiplier', 1.0)

# Thermal mode multiplier
thermal_multiplier = {
TradingMode.OPTIMAL_AGGRESSIVE: 1.5,
TradingMode.BALANCED_CONSISTENT: 1.0,
TradingMode.EFFICIENT_CONSERVATIVE: 0.7,
TradingMode.THROTTLE_SAFETY: 0.3,
TradingMode.CRITICAL_HALT: 0.0
}.get(thermal_mode, 1.0)

position_size = base_position * position_multiplier * thermal_multiplier

return float(xp.clip(position_size, 0.0, base_position))

except Exception as e:
logger.warning(f"⚠️ Failed to calculate position size: {e}")
return 0.0

def update_ghost_basket(self, basket_id: str, signal: TradingSignal) -> GhostBasket:
"""Update ghost basket with new trading signal."""
try:
if basket_id not in self.ghost_baskets:
# Create new ghost basket
self.ghost_baskets[basket_id] = GhostBasket(
basket_id=basket_id,
strategy_matrices=[],
profit_matrices=[],
total_value=0.0,
total_pnl=0.0,
risk_metrics={},
last_update=int(time.time() * 1000)
)

basket = self.ghost_baskets[basket_id]

# Update basket with new signal
if signal.signal_type in ['buy', 'sell']:
# Create profit matrix for this signal
profit_matrix = ProfitMatrix(
matrix_id=f"profit_{len(basket.profit_matrices)}",
entry_price=signal.price,
current_price=signal.price,
position_size=signal.amount,
unrealized_pnl=0.0,
tensor_profit_score=signal.tensor_score,
profit_target=signal.price * 1.001,  # 0.1% profit target
stop_loss=signal.price * 0.999,      # 0.1% stop loss
confidence=signal.confidence,
bit_phase=signal.bit_phase,
thermal_state=signal.metadata.get('thermal_state', 65.0)
)

basket.profit_matrices.append(profit_matrix)
basket.total_value += signal.amount * signal.price
basket.last_update = int(time.time() * 1000)

# Update risk metrics
basket.risk_metrics = self._calculate_basket_risk_metrics(basket)

logger.info(f"✅ Updated ghost basket: {basket_id}")
return basket

except Exception as e:
logger.error(f"❌ Failed to update ghost basket: {e}")
raise

def _calculate_basket_risk_metrics(self, basket: GhostBasket) -> Dict[str, float]:
"""Calculate risk metrics for ghost basket."""
try:
if not basket.profit_matrices:
return {}

# Calculate basic metrics
total_value = basket.total_value
total_pnl = basket.total_pnl

# Calculate volatility
if len(basket.profit_matrices) > 1:
pnl_values = [pm.unrealized_pnl for pm in basket.profit_matrices]
volatility = float(xp.std(pnl_values)) if len(pnl_values) > 1 else 0.0
else:
volatility = 0.0

# Calculate Sharpe ratio (simplified)
avg_pnl = total_pnl / len(basket.profit_matrices) if basket.profit_matrices else 0.0
sharpe_ratio = avg_pnl / volatility if volatility > 0 else 0.0

return {
'total_value': total_value,
'total_pnl': total_pnl,
'volatility': volatility,
'sharpe_ratio': sharpe_ratio,
'position_count': len(basket.profit_matrices),
'avg_confidence': float(xp.mean([pm.confidence for pm in basket.profit_matrices]))
}

except Exception as e:
logger.warning(f"⚠️ Failed to calculate basket risk metrics: {e}")
return {}

def get_trading_summary(self) -> Dict[str, Any]:
"""Get comprehensive trading summary."""
try:
return {
'total_signals': len(self.trading_signals),
'recent_signals': len([s for s in self.trading_signals if s.metadata.get('timestamp', 0) > time.time() - 3600]),
'ghost_baskets': len(self.ghost_baskets),
'strategy_matrices': len(self.strategy_matrices),
'profit_matrices': sum(len(b.profit_matrices) for b in self.ghost_baskets.values()),
'price_history_size': len(self.price_history),
'thermal_state': self.price_history[-1].thermal_state if self.price_history else 65.0,
'backend': _backend
}

except Exception as e:
logger.error(f"❌ Failed to get trading summary: {e}")
return {"error": str(e)}


# Singleton instance for global use
btc_trading_engine = BTCTradingEngine()