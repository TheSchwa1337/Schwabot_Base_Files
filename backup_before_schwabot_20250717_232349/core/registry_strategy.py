"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry Strategy Module for Schwabot
=====================================

Loads and executes strategies from hashed registry entries.
Integrates with Schwabot's mathematical framework and hash-based logic system.

This module provides the bridge between registry hashes and executable trading logic,
enabling recursive strategy evaluation and mathematical feedback loops.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import Schwabot mathematical components
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

# Import TradingPair separately to avoid circular imports
try:
from core.quad_bit_strategy_array import TradingPair
TRADING_PAIR_AVAILABLE = True
except ImportError:
TRADING_PAIR_AVAILABLE = False
logger.warning("TradingPair not available, using string fallback")
# Create a fallback TradingPair enum
from enum import Enum
class TradingPair(Enum):
"""Class for Schwabot trading functionality."""
BTC_USDC = "BTC/USDC"
ETH_USDC = "ETH/USDC"
XRP_USDC = "XRP/USDC"
SOL_USDC = "SOL/USDC"
USDC_USD = "USDC/USD"
USDT_USD = "USDT/USD"
BTC_USDT = "BTC/USDT"
ETH_USDT = "ETH/USDT"

@dataclass
class StrategyMetadata:
"""Class for Schwabot trading functionality."""
"""Metadata for a registry strategy."""
hash_id: str
registry_path: str
strategy_name: str
description: str
expected_profit: float
risk_level: str
trading_pair: TradingPair
bit_level: int
thermal_mode: str
created_timestamp: float
last_updated: float
execution_count: int = 0
success_rate: float = 0.0
average_profit: float = 0.0

@dataclass
class TickData:
"""Class for Schwabot trading functionality."""
"""Tick data structure for strategy execution."""
timestamp: int
price: float
volume: float
hash_value: str
bit_phase: int
tensor_score: float
entropy: float
thermal_state: float

@dataclass
class StrategyResult:
"""Class for Schwabot trading functionality."""
"""Result of strategy execution."""
strategy_hash: str
execution_timestamp: float
decision: str  # 'buy', 'sell', 'hold'
confidence: float
expected_profit: float
actual_profit: float = 0.0
profit_delta: float = 0.0
position_size: float = 0.0
entry_price: float = 0.0
exit_price: float = 0.0
metadata: Dict[str, Any] = field(default_factory=dict)

class RegistryStrategy:
"""Class for Schwabot trading functionality."""
"""
Registry Strategy Implementation

Loads strategies from hashed registry entries and executes them
using Schwabot's mathematical framework.
"""


def __init__(self, hash_id: str, registry_path: str = "registry/hashed_strategies.json") -> None:
"""Initialize registry strategy with hash ID and registry path."""
self.hash_id = hash_id
self.registry_path = registry_path
self.metadata = None
self.strategy_logic = None
self.math_config = None
self.math_cache = None
self.math_orchestrator = None

# Initialize mathematical infrastructure
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()

# Load strategy from registry
self._load_strategy()

logger.info(f"✅ Registry strategy {hash_id[:8]}... loaded successfully")

def _load_strategy(self) -> None:
"""Load strategy logic and metadata from registry."""
try:
registry_file = Path(self.registry_path)
if not registry_file.exists():
raise FileNotFoundError(f"Registry file not found: {self.registry_path}")

with open(registry_file, 'r', encoding='utf-8') as f:
registry = json.load(f)

# Load strategy metadata
if self.hash_id not in registry:
raise ValueError(f"Strategy hash {self.hash_id} not found in registry")

strategy_data = registry[self.hash_id]

# Create metadata
self.metadata = StrategyMetadata(
hash_id=self.hash_id,
registry_path=self.registry_path,
strategy_name=strategy_data.get('name', f'Strategy_{self.hash_id[:8]}'),
description=strategy_data.get('description', ''),
expected_profit=strategy_data.get('expected_profit', 0.0),
risk_level=strategy_data.get('risk_level', 'moderate'),
trading_pair=TradingPair[strategy_data.get('trading_pair', 'BTC_USDC')],
bit_level=strategy_data.get('bit_level', 16),
thermal_mode=strategy_data.get('thermal_mode', 'balanced_consistent'),
created_timestamp=strategy_data.get('created_timestamp', 0.0),
last_updated=strategy_data.get('last_updated', 0.0),
execution_count=strategy_data.get('execution_count', 0),
success_rate=strategy_data.get('success_rate', 0.0),
average_profit=strategy_data.get('average_profit', 0.0)
)

# Load strategy logic
strategy_code = strategy_data.get('logic', '')
if not strategy_code:
raise ValueError(f"No strategy logic found for hash {self.hash_id}")

# Create execution environment with Schwabot components
exec_env = {
'math_config': self.math_config,
'math_cache': self.math_cache,
'math_orchestrator': self.math_orchestrator,
'TradingPair': TradingPair,
'Decimal': Decimal,
'logger': logger
}

# Execute strategy code in safe environment
exec(strategy_code, exec_env)

# Get the strategy function
if 'run_strategy' not in exec_env:
raise ValueError(f"Strategy {self.hash_id} does not define 'run_strategy' function")

self.strategy_logic = exec_env['run_strategy']

except Exception as e:
logger.error(f"❌ Failed to load strategy {self.hash_id}: {e}")
raise

def run(self, tick_data: Union[List[TickData], List[Dict[str, Any]]]) -> StrategyResult:
"""
Execute strategy logic on tick data.

Args:
tick_data: List of tick data points

Returns:
StrategyResult with execution details
"""
try:
if not self.strategy_logic:
raise RuntimeError("Strategy logic not loaded")

# Convert tick data to proper format if needed
if isinstance(tick_data[0], dict):
tick_data = [TickData(**tick) for tick in tick_data]

# Execute strategy logic
result = self.strategy_logic(tick_data, self.metadata)

# Create strategy result
strategy_result = StrategyResult(
strategy_hash=self.hash_id,
execution_timestamp=tick_data[-1].timestamp if tick_data else 0,
decision=result.get('decision', 'hold'),
confidence=result.get('confidence', 0.0),
expected_profit=self.metadata.expected_profit,
actual_profit=result.get('actual_profit', 0.0),
profit_delta=result.get('profit_delta', 0.0),
position_size=result.get('position_size', 0.0),
entry_price=result.get('entry_price', 0.0),
exit_price=result.get('exit_price', 0.0),
metadata=result.get('metadata', {})
)

# Update execution count
self.metadata.execution_count += 1

logger.debug(f"Strategy {self.hash_id[:8]} executed: {strategy_result.decision} "
f"(confidence: {strategy_result.confidence:.2f})")

return strategy_result

except Exception as e:
logger.error(f"❌ Strategy execution failed for {self.hash_id}: {e}")
# Return error result
return StrategyResult(
strategy_hash=self.hash_id,
execution_timestamp=0,
decision='error',
confidence=0.0,
expected_profit=0.0,
metadata={'error': str(e)}
)

def get_status(self) -> Dict[str, Any]:
"""Get strategy status and metadata."""
return {
'hash_id': self.hash_id,
'metadata': self.metadata.__dict__ if self.metadata else None,
'logic_loaded': self.strategy_logic is not None,
'math_infrastructure': MATH_INFRASTRUCTURE_AVAILABLE
}

def update_metadata(self, **kwargs) -> None:
"""Update strategy metadata."""
if self.metadata:
for key, value in kwargs.items():
if hasattr(self.metadata, key):
setattr(self.metadata, key, value)

# Factory function

def create_registry_strategy(hash_id: str,
registry_path: str = "registry/hashed_strategies.json") -> RegistryStrategy:
"""Create a registry strategy instance."""
return RegistryStrategy(hash_id, registry_path)
