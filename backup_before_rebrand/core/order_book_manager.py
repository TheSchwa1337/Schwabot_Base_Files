"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Book Manager Module
===========================
Provides order book management functionality for the Schwabot trading system.

Mathematical Core:
O_t^n = OrderBook_{[t-n, t]} = {levels_1 ... levels_n}
- Used for calculating VWAP, MidPrice, etc.
- Maintains clean state of current L1-L3 book from WebSocket feeds
- Ensures time-synced snapshots

This module maintains the order book state and provides clean snapshots
for analysis and trading decisions.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from collections import deque

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
# Lazy import to avoid circular dependency
# from core.unified_mathematical_bridge import UnifiedMathematicalBridge
from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Mathematical infrastructure not available - using fallback")

def _get_unified_mathematical_bridge():
"""Lazy import to avoid circular dependency."""
try:
from core.unified_mathematical_bridge import UnifiedMathematicalBridge
return UnifiedMathematicalBridge
except ImportError:
logger.warning("UnifiedMathematicalBridge not available due to circular import")
return None

class OrderBookLevel(Enum):
"""Class for Schwabot trading functionality."""
"""Order book depth levels."""
L1 = "L1"  # Best bid/ask
L2 = "L2"  # Top 5 levels
L3 = "L3"  # Full depth

class OrderBookState(Enum):
"""Class for Schwabot trading functionality."""
"""Order book state."""
SYNCED = "synced"
UPDATING = "updating"
STALE = "stale"
ERROR = "error"

@dataclass
class OrderBookEntry:
"""Class for Schwabot trading functionality."""
"""Single order book entry."""
price: float
quantity: float
side: str  # 'bid' or 'ask'
timestamp: float = field(default_factory=time.time)
sequence_id: Optional[int] = None

@dataclass
class OrderBookSnapshot:
"""Class for Schwabot trading functionality."""
"""Complete order book snapshot."""
symbol: str
timestamp: float
sequence_id: Optional[int]
bids: List[OrderBookEntry]
asks: List[OrderBookEntry]
state: OrderBookState
mathematical_analysis: Dict[str, Any] = field(default_factory=dict)

# Calculated values
mid_price: float = 0.0
spread: float = 0.0
vwap: float = 0.0
mathematical_signature: str = ""

@dataclass
class OrderBookManagerConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for order book manager."""
enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False
max_snapshots: int = 100  # Maximum snapshots to keep in memory
snapshot_interval: float = 1.0  # Seconds between snapshots
stale_threshold: float = 5.0  # Seconds before snapshot is considered stale
mathematical_analysis_enabled: bool = True
depth_levels: OrderBookLevel = OrderBookLevel.L3

class OrderBookManager:
"""Class for Schwabot trading functionality."""
"""
Order Book Manager System

Implements order book state management:
O_t^n = OrderBook_{[t-n, t]} = {levels_1 ... levels_n}

Maintains clean state of current L1-L3 book from WebSocket feeds,
ensures time-synced snapshots, and provides calculated metrics.
"""


def __init__(self, config: Optional[OrderBookManagerConfig] = None) -> None:
"""Initialize the order book manager system."""
self.config = config or OrderBookManagerConfig()
self.logger = logging.getLogger(__name__)

# Order book data
self.order_books: Dict[str, Dict[str, List[OrderBookEntry]]] = {}  # symbol -> {bids, asks}
self.snapshots: Dict[str, deque] = {}  # symbol -> deque of snapshots
self.sequence_ids: Dict[str, int] = {}  # symbol -> last sequence ID
self.last_update: Dict[str, float] = {}  # symbol -> last update timestamp

# Mathematical infrastructure
if MATH_INFRASTRUCTURE_AVAILABLE:
UnifiedMathematicalBridgeClass = _get_unified_mathematical_bridge()
if UnifiedMathematicalBridgeClass:
self.math_bridge = UnifiedMathematicalBridgeClass()
self.math_integration = UnifiedMathematicalIntegrationMethods()
self.math_monitor = UnifiedMathematicalPerformanceMonitor()
else:
self.math_bridge = None
self.math_integration = None
self.math_monitor = None

# Performance tracking
self.performance_metrics = {
'updates_processed': 0,
'snapshots_created': 0,
'mathematical_analyses': 0,
'average_processing_time': 0.0
}

# System state
self.initialized = False
self.active = False

self._initialize_system()

def _initialize_system(self) -> None:
"""Initialize the order book manager system."""
try:
self.logger.info("Initializing Order Book Manager System")
self.initialized = True
self.logger.info("✅ Order Book Manager System initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing Order Book Manager System: {e}")
self.initialized = False


def update_order_book(self, symbol: str, side: str, updates: List[List[Union[str, float]]], -> None
sequence_id: Optional[int] = None) -> bool:
"""Update order book with new data."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
start_time = time.time()

# Initialize order book for symbol if not exists
if symbol not in self.order_books:
self.order_books[symbol] = {'bids': [], 'asks': []}
self.snapshots[symbol] = deque(maxlen=self.config.max_snapshots)
self.sequence_ids[symbol] = 0
self.last_update[symbol] = time.time()

# Validate sequence ID
if sequence_id is not None:
if sequence_id <= self.sequence_ids[symbol]:
self.logger.warning(f"Out of order sequence ID for {symbol}: {sequence_id} <= {self.sequence_ids[symbol]}")
return False
self.sequence_ids[symbol] = sequence_id

# Update order book entries
entries = []
for update in updates:
if len(update) >= 2:
price = float(update[0])
quantity = float(update[1])

if quantity > 0:
entry = OrderBookEntry(
price=price,
quantity=quantity,
side=side,
timestamp=time.time(),
sequence_id=sequence_id
)
entries.append(entry)

# Apply updates
self._apply_updates(symbol, side, entries)

# Update timestamp
self.last_update[symbol] = time.time()

# Create snapshot if needed
if self._should_create_snapshot(symbol):
self._create_snapshot(symbol)

# Update performance metrics
processing_time = time.time() - start_time
self.performance_metrics['updates_processed'] += 1

# Update average processing time
current_avg = self.performance_metrics['average_processing_time']
total_updates = self.performance_metrics['updates_processed']
self.performance_metrics['average_processing_time'] = (
(current_avg * (total_updates - 1) + processing_time) / total_updates
)

return True

except Exception as e:
self.logger.error(f"❌ Error updating order book for {symbol}: {e}")
return False

def _apply_updates(self, symbol: str, side: str, entries: List[OrderBookEntry]) -> None:
"""Apply updates to order book."""
try:
order_book = self.order_books[symbol]
side_key = side.lower()

if side_key not in order_book:
order_book[side_key] = []

# Clear existing entries for this side
order_book[side_key] = []

# Add new entries
order_book[side_key].extend(entries)

# Sort by price (bids descending, asks ascending)
if side_key == 'bids':
order_book[side_key].sort(key=lambda x: x.price, reverse=True)
else:
order_book[side_key].sort(key=lambda x: x.price)

# Limit depth based on configuration
max_levels = self._get_max_levels()
order_book[side_key] = order_book[side_key][:max_levels]

except Exception as e:
self.logger.error(f"❌ Error applying updates: {e}")

def _get_max_levels(self) -> int:
"""Get maximum levels based on configuration."""
if self.config.depth_levels == OrderBookLevel.L1:
return 1
elif self.config.depth_levels == OrderBookLevel.L2:
return 5
else:  # L3
return 100

def _should_create_snapshot(self, symbol: str) -> bool:
"""Check if we should create a new snapshot."""
try:
if symbol not in self.snapshots:
return True

snapshots = self.snapshots[symbol]
if not snapshots:
return True

last_snapshot = snapshots[-1]
time_since_last = time.time() - last_snapshot.timestamp

return time_since_last >= self.config.snapshot_interval

except Exception as e:
self.logger.error(f"❌ Error checking snapshot creation: {e}")
return True

def _create_snapshot(self, symbol: str) -> None:
"""Create a new order book snapshot."""
try:
order_book = self.order_books.get(symbol)
if not order_book:
return

# Determine state
time_since_update = time.time() - self.last_update.get(symbol, 0)
if time_since_update > self.config.stale_threshold:
state = OrderBookState.STALE
else:
state = OrderBookState.SYNCED

# Create snapshot
snapshot = OrderBookSnapshot(
symbol=symbol,
timestamp=time.time(),
sequence_id=self.sequence_ids.get(symbol),
bids=order_book.get('bids', []).copy(),
asks=order_book.get('asks', []).copy(),
state=state
)

# Calculate metrics
self._calculate_snapshot_metrics(snapshot)

# Perform mathematical analysis
if self.config.mathematical_analysis_enabled:
self._perform_mathematical_analysis(snapshot)

# Store snapshot
self.snapshots[symbol].append(snapshot)

# Update performance metrics
self.performance_metrics['snapshots_created'] += 1
self.performance_metrics['mathematical_analyses'] += 1

except Exception as e:
self.logger.error(f"❌ Error creating snapshot for {symbol}: {e}")

def _calculate_snapshot_metrics(self, snapshot: OrderBookSnapshot) -> None:
"""Calculate metrics for snapshot."""
try:
bids = snapshot.bids
asks = snapshot.asks

# Calculate mid price
if bids and asks:
best_bid = max(bid.price for bid in bids)
best_ask = min(ask.price for ask in asks)
snapshot.mid_price = (best_bid + best_ask) / 2
snapshot.spread = best_ask - best_bid
elif bids:
snapshot.mid_price = max(bid.price for bid in bids)
snapshot.spread = 0.0
elif asks:
snapshot.mid_price = min(ask.price for ask in asks)
snapshot.spread = 0.0

# Calculate VWAP
total_volume = 0.0
weighted_sum = 0.0

for entry in bids + asks:
total_volume += entry.quantity
weighted_sum += entry.price * entry.quantity

if total_volume > 0:
snapshot.vwap = weighted_sum / total_volume
else:
snapshot.vwap = snapshot.mid_price

# Create mathematical signature
snapshot.mathematical_signature = self._create_snapshot_signature(snapshot)

except Exception as e:
self.logger.error(f"❌ Error calculating snapshot metrics: {e}")

def _perform_mathematical_analysis(self, snapshot: OrderBookSnapshot) -> None:
"""Perform mathematical analysis on snapshot."""
try:
if not self.math_bridge:
return

# Prepare data for mathematical analysis
order_book_data = {
'symbol': snapshot.symbol,
'timestamp': snapshot.timestamp,
'mid_price': snapshot.mid_price,
'spread': snapshot.spread,
'vwap': snapshot.vwap,
'bid_count': len(snapshot.bids),
'ask_count': len(snapshot.asks),
'total_bid_quantity': sum(bid.quantity for bid in snapshot.bids),
'total_ask_quantity': sum(ask.quantity for ask in snapshot.asks),
'bid_prices': [bid.price for bid in snapshot.bids],
'ask_prices': [ask.price for ask in snapshot.asks],
'bid_quantities': [bid.quantity for bid in snapshot.bids],
'ask_quantities': [ask.quantity for ask in snapshot.asks]
}

# Perform mathematical integration
result = self.math_bridge.integrate_all_mathematical_systems(
order_book_data, {}
)

snapshot.mathematical_analysis = {
'confidence': result.overall_confidence,
'connections': len(result.connections),
'performance_metrics': result.performance_metrics,
'mathematical_signature': result.mathematical_signature
}

except Exception as e:
self.logger.error(f"❌ Error performing mathematical analysis: {e}")

def _create_snapshot_signature(self, snapshot: OrderBookSnapshot) -> str:
"""Create mathematical signature for snapshot."""
try:
signature_components = [
f"MP:{snapshot.mid_price:.6f}",
f"SP:{snapshot.spread:.6f}",
f"VWAP:{snapshot.vwap:.6f}",
f"B:{len(snapshot.bids)}",
f"A:{len(snapshot.asks)}",
f"TQ:{sum(bid.quantity for bid in snapshot.bids) + sum(ask.quantity for ask in snapshot.asks):.6f}"
]
return "|".join(signature_components)
except Exception as e:
self.logger.error(f"❌ Error creating snapshot signature: {e}")
return ""

def get_order_book(self, symbol: str, level: OrderBookLevel = OrderBookLevel.L3) -> Optional[Dict[str, Any]]:
"""Get current order book for a symbol."""
try:
order_book = self.order_books.get(symbol)
if not order_book:
return None

# Get appropriate depth
max_levels = self._get_max_levels_for_level(level)

bids = order_book.get('bids', [])[:max_levels]
asks = order_book.get('asks', [])[:max_levels]

return {
'symbol': symbol,
'timestamp': self.last_update.get(symbol, 0),
'sequence_id': self.sequence_ids.get(symbol),
'bids': [[entry.price, entry.quantity] for entry in bids],
'asks': [[entry.price, entry.quantity] for entry in asks],
'state': self._get_order_book_state(symbol).value
}

except Exception as e:
self.logger.error(f"❌ Error getting order book for {symbol}: {e}")
return None

def _get_max_levels_for_level(self, level: OrderBookLevel) -> int:
"""Get maximum levels for specific depth level."""
if level == OrderBookLevel.L1:
return 1
elif level == OrderBookLevel.L2:
return 5
else:  # L3
return 100

def _get_order_book_state(self, symbol: str) -> OrderBookState:
"""Get order book state for symbol."""
try:
time_since_update = time.time() - self.last_update.get(symbol, 0)
if time_since_update > self.config.stale_threshold:
return OrderBookState.STALE
else:
return OrderBookState.SYNCED
except Exception as e:
self.logger.error(f"❌ Error getting order book state: {e}")
return OrderBookState.ERROR

def get_latest_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
"""Get latest snapshot for a symbol."""
try:
snapshots = self.snapshots.get(symbol)
if not snapshots:
return None

return snapshots[-1]
except Exception as e:
self.logger.error(f"❌ Error getting latest snapshot for {symbol}: {e}")
return None

def get_snapshot_history(self, symbol: str, count: int = 10) -> List[OrderBookSnapshot]:
"""Get snapshot history for a symbol."""
try:
snapshots = self.snapshots.get(symbol, deque())
return list(snapshots)[-count:]
except Exception as e:
self.logger.error(f"❌ Error getting snapshot history for {symbol}: {e}")
return []

def get_all_symbols(self) -> List[str]:
"""Get all tracked symbols."""
return list(self.order_books.keys())

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get system performance metrics."""
return self.performance_metrics.copy()

def clear_order_book(self, symbol: str) -> bool:
"""Clear order book for a symbol."""
try:
if symbol in self.order_books:
del self.order_books[symbol]
if symbol in self.snapshots:
del self.snapshots[symbol]
if symbol in self.sequence_ids:
del self.sequence_ids[symbol]
if symbol in self.last_update:
del self.last_update[symbol]

self.logger.info(f"✅ Cleared order book for {symbol}")
return True

except Exception as e:
self.logger.error(f"❌ Error clearing order book for {symbol}: {e}")
return False

def activate(self) -> bool:
"""Activate the system."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True
self.logger.info("✅ Order Book Manager System activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating Order Book Manager System: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the system."""
try:
self.active = False
self.logger.info("✅ Order Book Manager System deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating Order Book Manager System: {e}")
return False

def get_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
'active': self.active,
'initialized': self.initialized,
'symbols_tracked': len(self.order_books),
'total_snapshots': sum(len(snapshots) for snapshots in self.snapshots.values()),
'performance_metrics': self.performance_metrics,
'config': {
'enabled': self.config.enabled,
'max_snapshots': self.config.max_snapshots,
'snapshot_interval': self.config.snapshot_interval,
'depth_levels': self.config.depth_levels.value,
'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled
}
}

def create_order_book_manager(config: Optional[OrderBookManagerConfig] = None) -> OrderBookManager:
"""Factory function to create OrderBookManager instance."""
return OrderBookManager(config)

def main():
"""Main function for testing."""
# Create configuration
config = OrderBookManagerConfig(
enabled=True,
debug=True,
max_snapshots=50,
snapshot_interval=1.0,
depth_levels=OrderBookLevel.L2,
mathematical_analysis_enabled=True
)

# Create manager
manager = create_order_book_manager(config)

# Activate system
manager.activate()

# Sample order book updates
symbol = "BTCUSDT"

# Update bids
bid_updates = [
['50000.0', '1.5'],
['49999.0', '2.0'],
['49998.0', '1.8'],
['49997.0', '0.5'],
['49996.0', '0.3']
]
manager.update_order_book(symbol, 'bids', bid_updates, sequence_id=1)

# Update asks
ask_updates = [
['50001.0', '1.2'],
['50002.0', '2.5'],
['50003.0', '1.0'],
['50004.0', '0.8'],
['50005.0', '0.6']
]
manager.update_order_book(symbol, 'asks', ask_updates, sequence_id=2)

# Get order book
order_book = manager.get_order_book(symbol, OrderBookLevel.L2)
print(f"Order Book: {json.dumps(order_book, indent=2)}")

# Get latest snapshot
snapshot = manager.get_latest_snapshot(symbol)
if snapshot:
print(f"Latest Snapshot:")
print(f"  Mid Price: ${snapshot.mid_price}")
print(f"  Spread: ${snapshot.spread}")
print(f"  VWAP: ${snapshot.vwap}")
print(f"  State: {snapshot.state.value}")

# Get status
status = manager.get_status()
print(f"System Status: {status}")

# Deactivate system
manager.deactivate()

if __name__ == "__main__":
main()
