"""Module for Schwabot trading system."""

#!/usr/bin/env python3
"""
Trade Registry - Canonical Trade Hash Registry
=============================================

The main trade registry that serves as the single source of truth for all executed trades.
Specialized registries reference this main registry by trade hash/ID to avoid redundancy.

Features:
- Canonical trade storage with SHA-256 hashing
- Integration points for specialized registries
- Performance tracking and analytics
- Backtesting support with full trade history
- Registry linkage management
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set
from decimal import Decimal
from datetime import datetime

# Import centralized hash configuration
from core.hash_config_manager import generate_hash_from_string

logger = logging.getLogger(__name__)

@dataclass
class TradeEntry:
"""Class for Schwabot trading functionality."""
"""Canonical trade entry with full metadata."""
trade_hash: str
timestamp: float
symbol: str
action: str  # 'buy' or 'sell'
entry_price: float
exit_price: Optional[float] = None
amount: float = 0.0
fees: float = 0.0
profit_usd: Optional[float] = None
profit_percentage: Optional[float] = None

# Strategy and signal metadata
strategy_id: Optional[str] = None
signal_strength: Optional[float] = None
confidence: Optional[float] = None

# Mathematical context
chrono_resonance: Optional[float] = None
temporal_warp: Optional[float] = None
math_optimization: Optional[Dict[str, float]] = None

# Market context
market_conditions: Optional[Dict[str, Any]] = None
volatility: Optional[float] = None
volume: Optional[float] = None

# Registry linkage
linked_registries: Set[str] = field(default_factory=set)
specialized_hashes: Dict[str, str] = field(default_factory=dict)

# Performance tracking
execution_time: Optional[float] = None
slippage: Optional[float] = None
success: Optional[bool] = None

# Metadata
metadata: Dict[str, Any] = field(default_factory=dict)

class TradeRegistry:
"""Class for Schwabot trading functionality."""
"""Canonical trade registry - single source of truth for all trades."""


def __init__(self, registry_file: Optional[str] = None) -> None:
"""Initialize the canonical trade registry."""
self.registry_file = registry_file or "data/canonical_trade_registry.json"
self.trades: Dict[str, TradeEntry] = {}
self.hash_index: Dict[str, str] = {}  # hash -> trade_hash
self.symbol_index: Dict[str, List[str]] = {}  # symbol -> trade_hashes
self.strategy_index: Dict[str, List[str]] = {}  # strategy_id -> trade_hashes
self.timestamp_index: List[str] = []  # ordered by timestamp

# Performance tracking
self.total_trades = 0
self.total_profit = 0.0
self.successful_trades = 0

# Registry linkage tracking
# registry_name -> {trade_hash -> specialized_hash}
self.specialized_registries: Dict[str, Dict[str, str]] = {}

self._load_registry()
logger.info(f"📊 Canonical Trade Registry initialized with {len(self.trades)} trades")

def add_trade(self, trade_data: Dict[str, Any]) -> str:
"""Add a trade to the canonical registry and return its hash."""
try:
# Generate canonical trade hash
trade_hash = self._generate_trade_hash(trade_data)

# Create trade entry
trade_entry = TradeEntry(
trade_hash=trade_hash,
timestamp=trade_data.get('timestamp', time.time()),
symbol=trade_data.get('symbol', 'UNKNOWN'),
action=trade_data.get('action', 'unknown'),
entry_price=trade_data.get('entry_price', 0.0),
exit_price=trade_data.get('exit_price'),
amount=trade_data.get('amount', 0.0),
fees=trade_data.get('fees', 0.0),
profit_usd=trade_data.get('profit_usd'),
profit_percentage=trade_data.get('profit_percentage'),
strategy_id=trade_data.get('strategy_id'),
signal_strength=trade_data.get('signal_strength'),
confidence=trade_data.get('confidence'),
chrono_resonance=trade_data.get('chrono_resonance'),
temporal_warp=trade_data.get('temporal_warp'),
math_optimization=trade_data.get('math_optimization'),
market_conditions=trade_data.get('market_conditions'),
volatility=trade_data.get('volatility'),
volume=trade_data.get('volume'),
execution_time=trade_data.get('execution_time'),
slippage=trade_data.get('slippage'),
success=trade_data.get('success'),
metadata=trade_data.get('metadata', {})
)

# Store in registry
self.trades[trade_hash] = trade_entry

# Update indices
self._update_indices(trade_hash, trade_entry)

# Update performance metrics
self._update_performance_metrics(trade_entry)

# Save to file
self._save_registry()

logger.info(f"✅ Trade registered: {trade_hash[:8]}... | {trade_entry.symbol} {trade_entry.action}")
return trade_hash

except Exception as e:
logger.error(f"Error adding trade to registry: {e}")
raise

def link_specialized_registry(self, trade_hash: str, registry_name: str, specialized_hash: str) -> bool:
"""Link a trade to a specialized registry."""
if trade_hash not in self.trades:
logger.warning(f"Trade hash {trade_hash} not found in canonical registry")
return False

trade_entry = self.trades[trade_hash]
trade_entry.linked_registries.add(registry_name)
trade_entry.specialized_hashes[registry_name] = specialized_hash

# Track in specialized registries index
if registry_name not in self.specialized_registries:
self.specialized_registries[registry_name] = {}
self.specialized_registries[registry_name][trade_hash] = specialized_hash

logger.info(f"🔗 Linked {trade_hash[:8]}... to {registry_name}: {specialized_hash[:8]}...")
return True

def get_trade(self, trade_hash: str) -> Optional[TradeEntry]:
"""Get a trade entry by its hash."""
return self.trades.get(trade_hash)

def get_trades_by_symbol(self, symbol: str) -> List[TradeEntry]:
"""Get all trades for a specific symbol."""
trade_hashes = self.symbol_index.get(symbol, [])
return [self.trades[th] for th in trade_hashes if th in self.trades]

def get_trades_by_strategy(self, strategy_id: str) -> List[TradeEntry]:
"""Get all trades for a specific strategy."""
trade_hashes = self.strategy_index.get(strategy_id, [])
return [self.trades[th] for th in trade_hashes if th in self.trades]

def get_recent_trades(self, count: int = 10) -> List[TradeEntry]:
"""Get the most recent trades."""
recent_hashes = self.timestamp_index[-count:] if self.timestamp_index else []
return [self.trades[th] for th in recent_hashes if th in self.trades]

def get_profitable_trades(self, min_profit: float = 0.0) -> List[TradeEntry]:
"""Get all profitable trades above a minimum threshold."""
profitable = []
for trade in self.trades.values():
if trade.profit_usd and trade.profit_usd >= min_profit:
profitable.append(trade)
return sorted(profitable, key=lambda t: t.profit_usd or 0, reverse=True)

def get_performance_summary(self) -> Dict[str, Any]:
"""Get comprehensive performance summary."""
if not self.trades:
return {"error": "No trades in registry"}

total_trades = len(self.trades)
completed_trades = [t for t in self.trades.values() if t.success is not None]

if not completed_trades:
return {"total_trades": total_trades, "completed_trades": 0}

successful_trades = [t for t in completed_trades if t.success]
total_profit = sum(t.profit_usd or 0 for t in completed_trades)
avg_profit = total_profit / len(completed_trades) if completed_trades else 0

return {
"total_trades": total_trades,
"completed_trades": len(completed_trades),
"successful_trades": len(successful_trades),
"success_rate": len(successful_trades) / len(completed_trades) if completed_trades else 0,
"total_profit": total_profit,
"average_profit": avg_profit,
"best_trade": max(completed_trades, key=lambda t: t.profit_usd or 0).profit_usd if completed_trades else 0,
"worst_trade": min(completed_trades, key=lambda t: t.profit_usd or 0).profit_usd if completed_trades else 0,
"linked_registries": list(self.specialized_registries.keys())
}

def get_registry_linkage(self, trade_hash: str) -> Dict[str, str]:
"""Get all specialized registry linkages for a trade."""
trade = self.trades.get(trade_hash)
if not trade:
return {}
return trade.specialized_hashes.copy()

def _generate_trade_hash(self, trade_data: Dict[str, Any]) -> str:
"""Generate a unique hash for the trade."""
# Create a deterministic representation
hash_data = {
'symbol': trade_data.get('symbol', ''),
'action': trade_data.get('action', ''),
'entry_price': trade_data.get('entry_price', 0),
'timestamp': trade_data.get('timestamp', time.time()),
'strategy_id': trade_data.get('strategy_id', ''),
'amount': trade_data.get('amount', 0)
}

trade_json = json.dumps(hash_data, sort_keys=True)
return generate_hash_from_string(trade_json)

def _update_indices(self, trade_hash: str, trade_entry: TradeEntry) -> None:
"""Update all indices for the trade."""
# Hash index
self.hash_index[trade_hash] = trade_hash

# Symbol index
if trade_entry.symbol not in self.symbol_index:
self.symbol_index[trade_entry.symbol] = []
self.symbol_index[trade_entry.symbol].append(trade_hash)

# Strategy index
if trade_entry.strategy_id:
if trade_entry.strategy_id not in self.strategy_index:
self.strategy_index[trade_entry.strategy_id] = []
self.strategy_index[trade_entry.strategy_id].append(trade_hash)

# Timestamp index (maintain order)
self.timestamp_index.append(trade_hash)

def _update_performance_metrics(self, trade_entry: TradeEntry) -> None:
"""Update performance tracking metrics."""
self.total_trades += 1

if trade_entry.profit_usd:
self.total_profit += trade_entry.profit_usd

if trade_entry.success is not None:
if trade_entry.success:
self.successful_trades += 1

def _load_registry(self) -> None:
"""Load registry from file."""
try:
import os
if os.path.exists(self.registry_file):
with open(self.registry_file, 'r', encoding='utf-8') as f:
data = json.load(f)

# Load trades
for trade_data in data.get('trades', []):
trade_entry = TradeEntry(**trade_data)
self.trades[trade_entry.trade_hash] = trade_entry

# Rebuild indices
for trade_hash, trade_entry in self.trades.items():
self._update_indices(trade_hash, trade_entry)

# Load specialized registry linkages
self.specialized_registries = data.get('specialized_registries', {})

logger.info(f"📊 Loaded {len(self.trades)} trades from registry file")

except Exception as e:
logger.warning(f"Could not load registry file: {e}")

def _save_registry(self) -> None:
"""Save registry to file."""
try:
import os
os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)

data = {
'trades': [trade.__dict__ for trade in self.trades.values()],
'specialized_registries': self.specialized_registries,
'metadata': {
'total_trades': self.total_trades,
'total_profit': self.total_profit,
'successful_trades': self.successful_trades,
'last_updated': datetime.now().isoformat()
}
}

with open(self.registry_file, 'w', encoding='utf-8') as f:
json.dump(data, f, indent=2, default=str)

except Exception as e:
logger.error(f"Error saving registry: {e}")

def clear(self) -> None:
"""Clear the registry (use with caution)."""
self.trades.clear()
self.hash_index.clear()
self.symbol_index.clear()
self.strategy_index.clear()
self.timestamp_index.clear()
self.specialized_registries.clear()
self.total_trades = 0
self.total_profit = 0.0
self.successful_trades = 0
logger.warning("🗑️ Canonical trade registry cleared")

# Global instance for easy access
canonical_trade_registry = TradeRegistry()