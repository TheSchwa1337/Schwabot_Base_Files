"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Book Analyzer Module
===========================
Provides order book analyzer functionality for the Schwabot trading system.

Mathematical Core:
Imbalance(t) = (ΣBids_t - ΣAsks_t) / (ΣBids_t + ΣAsks_t)
- Signal Input: Used to bias strategy towards aggressiveness or stealth
- Detects buy/sell walls, liquidity cliffs, and depth imbalances

This module analyzes bid-ask spread, depth imbalance, wall formations, and
liquidity cliffs to provide trading signals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

class WallType(Enum):
"""Class for Schwabot trading functionality."""
"""Types of order book walls."""
BUY_WALL = "buy_wall"
SELL_WALL = "sell_wall"
LIQUIDITY_CLIFF = "liquidity_cliff"
SUPPORT_LEVEL = "support_level"
RESISTANCE_LEVEL = "resistance_level"

class MarketStructure(Enum):
"""Class for Schwabot trading functionality."""
"""Market structure classification."""
BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"
VOLATILE = "volatile"
CONSOLIDATING = "consolidating"

@dataclass
class OrderBookLevel:
"""Class for Schwabot trading functionality."""
"""Single order book level."""
price: float
quantity: float
side: str  # 'bid' or 'ask'
timestamp: float = field(default_factory=time.time)

@dataclass
class OrderBookWall:
"""Class for Schwabot trading functionality."""
"""Detected order book wall."""
wall_type: WallType
price_level: float
total_quantity: float
strength: float  # 0.0 to 1.0
confidence: float  # 0.0 to 1.0
mathematical_signature: str = ""
timestamp: float = field(default_factory=time.time)

@dataclass
class LiquidityAnalysis:
"""Class for Schwabot trading functionality."""
"""Liquidity analysis results."""
bid_liquidity: float
ask_liquidity: float
imbalance_ratio: float
spread: float
depth_score: float
mathematical_signature: str = ""

@dataclass
class OrderBookSnapshot:
"""Class for Schwabot trading functionality."""
"""Complete order book snapshot."""
symbol: str
timestamp: float
bids: List[OrderBookLevel]
asks: List[OrderBookLevel]
mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
market_structure: MarketStructure = MarketStructure.NEUTRAL

@dataclass
class OrderBookAnalyzerConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for order book analyzer."""
enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False
wall_detection_threshold: float = 0.1  # Minimum wall strength
imbalance_threshold: float = 0.05  # Minimum imbalance to consider
max_depth_levels: int = 20
mathematical_analysis_enabled: bool = True

@dataclass
class OrderBookAnalyzerMetrics:
"""Class for Schwabot trading functionality."""
"""Order book analyzer metrics."""
snapshots_analyzed: int = 0
walls_detected: int = 0
liquidity_analyses: int = 0
mathematical_analyses: int = 0
average_processing_time: float = 0.0
last_updated: float = 0.0

class OrderBookAnalyzer:
"""Class for Schwabot trading functionality."""
"""
Order Book Analyzer System

Implements mathematical analysis of order book data:
Imbalance(t) = (ΣBids_t - ΣAsks_t) / (ΣBids_t + ΣAsks_t)

Analyzes bid-ask spread, depth imbalance, wall formations, and
liquidity cliffs to provide trading signals.
"""


def __init__(self, config: Optional[OrderBookAnalyzerConfig] = None) -> None:
"""Initialize the order book analyzer system."""
self.config = config or OrderBookAnalyzerConfig()
self.logger = logging.getLogger(__name__)

# Order book data
self.order_book_snapshots: Dict[str, OrderBookSnapshot] = {}
self.wall_history: Dict[str, List[OrderBookWall]] = {}
self.liquidity_history: Dict[str, List[LiquidityAnalysis]] = {}

# Mathematical infrastructure
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()
else:
self.math_config = None
self.math_cache = None
self.math_orchestrator = None

# Performance tracking
self.metrics = OrderBookAnalyzerMetrics()

# System state
self.initialized = False
self.active = False

self._initialize_system()

def _initialize_system(self) -> None:
"""Initialize the order book analyzer system."""
try:
self.logger.info("Initializing Order Book Analyzer System")
self.initialized = True
self.logger.info("✅ Order Book Analyzer System initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing Order Book Analyzer System: {e}")
self.initialized = False

def analyze_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
"""Analyze order book data and return comprehensive analysis."""
if not self.initialized:
self.logger.error("System not initialized")
return {}

try:
start_time = time.time()

# Parse order book data
snapshot = self._parse_order_book_data(symbol, order_book_data)

# Perform mathematical analysis
mathematical_analysis = self._perform_mathematical_analysis(snapshot)

# Detect walls
walls = self._detect_walls(snapshot)

# Analyze liquidity
liquidity_analysis = self._analyze_liquidity(snapshot)

# Classify market structure
market_structure = self._classify_market_structure(snapshot, walls, liquidity_analysis)

# Update snapshot with analysis
snapshot.mathematical_analysis = mathematical_analysis
snapshot.market_structure = market_structure

# Store snapshot
self.order_book_snapshots[symbol] = snapshot

# Update history
if symbol not in self.wall_history:
self.wall_history[symbol] = []
self.wall_history[symbol].extend(walls)

if symbol not in self.liquidity_history:
self.liquidity_history[symbol] = []
self.liquidity_history[symbol].append(liquidity_analysis)

# Update metrics
processing_time = time.time() - start_time
self.metrics.snapshots_analyzed += 1
self.metrics.walls_detected += len(walls)
self.metrics.liquidity_analyses += 1
self.metrics.mathematical_analyses += 1

# Update average processing time
current_avg = self.metrics.average_processing_time
total_analyses = self.metrics.snapshots_analyzed
self.metrics.average_processing_time = (
(current_avg * (total_analyses - 1) + processing_time) / total_analyses
)

# Return analysis results
return {
'symbol': symbol,
'timestamp': snapshot.timestamp,
'market_structure': market_structure.value,
'walls': [self._wall_to_dict(wall) for wall in walls],
'liquidity': self._liquidity_to_dict(liquidity_analysis),
'mathematical_analysis': mathematical_analysis,
'processing_time': processing_time
}

except Exception as e:
self.logger.error(f"❌ Error analyzing order book for {symbol}: {e}")
return {}

def _parse_order_book_data(self, symbol: str, order_book_data: Dict[str, Any]) -> OrderBookSnapshot:
"""Parse order book data into structured format."""
try:
bids = []
asks = []

# Parse bids
if 'bids' in order_book_data:
for bid in order_book_data['bids'][:self.config.max_depth_levels]:
if len(bid) >= 2:
bids.append(OrderBookLevel(
price=float(bid[0]),
quantity=float(bid[1]),
side='bid'
))

# Parse asks
if 'asks' in order_book_data:
for ask in order_book_data['asks'][:self.config.max_depth_levels]:
if len(ask) >= 2:
asks.append(OrderBookLevel(
price=float(ask[0]),
quantity=float(ask[1]),
side='ask'
))

return OrderBookSnapshot(
symbol=symbol,
timestamp=time.time(),
bids=bids,
asks=asks
)

except Exception as e:
self.logger.error(f"❌ Error parsing order book data: {e}")
return OrderBookSnapshot(symbol=symbol, timestamp=time.time(), bids=[], asks=[])

def _detect_walls(self, snapshot: OrderBookSnapshot) -> List[OrderBookWall]:
"""Detect order book walls."""
try:
walls = []

# Detect buy walls
buy_walls = self._detect_buy_walls(snapshot.bids)
walls.extend(buy_walls)

# Detect sell walls
sell_walls = self._detect_sell_walls(snapshot.asks)
walls.extend(sell_walls)

# Detect liquidity cliffs
cliffs = self._detect_liquidity_cliffs(snapshot)
walls.extend(cliffs)

return walls

except Exception as e:
self.logger.error(f"❌ Error detecting walls: {e}")
return []

def _detect_buy_walls(self, bids: List[OrderBookLevel]) -> List[OrderBookWall]:
"""Detect buy walls in bid orders."""
try:
walls = []

if len(bids) < 3:
return walls

# Group consecutive price levels
current_wall_quantity = 0.0
current_wall_price = bids[0].price
wall_levels = 0

for i, bid in enumerate(bids):
# Check if price is close to previous (within 0.1%)
if abs(bid.price - current_wall_price) / current_wall_price < 0.001:
current_wall_quantity += bid.quantity
wall_levels += 1
else:
# Check if we have a significant wall
if wall_levels >= 3 and current_wall_quantity > 0:
# Calculate wall strength using mathematical infrastructure
wall_strength = self._calculate_wall_strength(
current_wall_quantity, wall_levels, current_wall_price
)

if wall_strength > self.config.wall_detection_threshold:
wall = OrderBookWall(
wall_type=WallType.BUY_WALL,
price_level=current_wall_price,
total_quantity=current_wall_quantity,
strength=wall_strength,
confidence=min(wall_strength * 1.2, 1.0),
mathematical_signature=self._create_wall_signature(
current_wall_price, current_wall_quantity,
wall_strength, "buy_wall"
)
)
walls.append(wall)

# Start new wall
current_wall_quantity = bid.quantity
current_wall_price = bid.price
wall_levels = 1

# Check final wall
if wall_levels >= 3 and current_wall_quantity > 0:
wall_strength = self._calculate_wall_strength(
current_wall_quantity, wall_levels, current_wall_price
)

if wall_strength > self.config.wall_detection_threshold:
wall = OrderBookWall(
wall_type=WallType.BUY_WALL,
price_level=current_wall_price,
total_quantity=current_wall_quantity,
strength=wall_strength,
confidence=min(wall_strength * 1.2, 1.0),
mathematical_signature=self._create_wall_signature(
current_wall_price, current_wall_quantity,
wall_strength, "buy_wall"
)
)
walls.append(wall)

return walls

except Exception as e:
self.logger.error(f"❌ Error detecting buy walls: {e}")
return []

def _detect_sell_walls(self, asks: List[OrderBookLevel]) -> List[OrderBookWall]:
"""Detect sell walls in ask orders."""
try:
walls = []

if len(asks) < 3:
return walls

# Group consecutive price levels
current_wall_quantity = 0.0
current_wall_price = asks[0].price
wall_levels = 0

for i, ask in enumerate(asks):
# Check if price is close to previous (within 0.1%)
if abs(ask.price - current_wall_price) / current_wall_price < 0.001:
current_wall_quantity += ask.quantity
wall_levels += 1
else:
# Check if we have a significant wall
if wall_levels >= 3 and current_wall_quantity > 0:
# Calculate wall strength using mathematical infrastructure
wall_strength = self._calculate_wall_strength(
current_wall_quantity, wall_levels, current_wall_price
)

if wall_strength > self.config.wall_detection_threshold:
wall = OrderBookWall(
wall_type=WallType.SELL_WALL,
price_level=current_wall_price,
total_quantity=current_wall_quantity,
strength=wall_strength,
confidence=min(wall_strength * 1.2, 1.0),
mathematical_signature=self._create_wall_signature(
current_wall_price, current_wall_quantity,
wall_strength, "sell_wall"
)
)
walls.append(wall)

# Start new wall
current_wall_quantity = ask.quantity
current_wall_price = ask.price
wall_levels = 1

# Check final wall
if wall_levels >= 3 and current_wall_quantity > 0:
wall_strength = self._calculate_wall_strength(
current_wall_quantity, wall_levels, current_wall_price
)

if wall_strength > self.config.wall_detection_threshold:
wall = OrderBookWall(
wall_type=WallType.SELL_WALL,
price_level=current_wall_price,
total_quantity=current_wall_quantity,
strength=wall_strength,
confidence=min(wall_strength * 1.2, 1.0),
mathematical_signature=self._create_wall_signature(
current_wall_price, current_wall_quantity,
wall_strength, "sell_wall"
)
)
walls.append(wall)

return walls

except Exception as e:
self.logger.error(f"❌ Error detecting sell walls: {e}")
return []

def _detect_liquidity_cliffs(self, snapshot: OrderBookSnapshot) -> List[OrderBookWall]:
"""Detect liquidity cliffs (sudden drops in liquidity)."""
try:
cliffs = []

# Analyze bid liquidity distribution
if len(snapshot.bids) > 5:
bid_quantities = [bid.quantity for bid in snapshot.bids]
bid_std = np.std(bid_quantities)
bid_mean = np.mean(bid_quantities)

# Find levels with significantly lower liquidity
for i, bid in enumerate(snapshot.bids):
if bid.quantity < (bid_mean - 2 * bid_std) and bid.quantity > 0:
cliff = OrderBookWall(
wall_type=WallType.LIQUIDITY_CLIFF,
price_level=bid.price,
total_quantity=bid.quantity,
strength=1.0 - (bid.quantity / bid_mean),
confidence=0.8,
mathematical_signature=self._create_wall_signature(
bid.price, bid.quantity,
1.0 - (bid.quantity / bid_mean), "liquidity_cliff"
)
)
cliffs.append(cliff)

# Analyze ask liquidity distribution
if len(snapshot.asks) > 5:
ask_quantities = [ask.quantity for ask in snapshot.asks]
ask_std = np.std(ask_quantities)
ask_mean = np.mean(ask_quantities)

# Find levels with significantly lower liquidity
for i, ask in enumerate(snapshot.asks):
if ask.quantity < (ask_mean - 2 * ask_std) and ask.quantity > 0:
cliff = OrderBookWall(
wall_type=WallType.LIQUIDITY_CLIFF,
price_level=ask.price,
total_quantity=ask.quantity,
strength=1.0 - (ask.quantity / ask_mean),
confidence=0.8,
mathematical_signature=self._create_wall_signature(
ask.price, ask.quantity,
1.0 - (ask.quantity / ask_mean), "liquidity_cliff"
)
)
cliffs.append(cliff)

return cliffs

except Exception as e:
self.logger.error(f"❌ Error detecting liquidity cliffs: {e}")
return []

def _analyze_liquidity(self, snapshot: OrderBookSnapshot) -> LiquidityAnalysis:
"""Analyze liquidity and calculate imbalance ratio."""
try:
# Calculate bid and ask liquidity
bid_liquidity = sum(bid.quantity for bid in snapshot.bids)
ask_liquidity = sum(ask.quantity for ask in snapshot.asks)

# Calculate imbalance ratio: (ΣBids - ΣAsks) / (ΣBids + ΣAsks)
total_liquidity = bid_liquidity + ask_liquidity
if total_liquidity > 0:
imbalance_ratio = (bid_liquidity - ask_liquidity) / total_liquidity
else:
imbalance_ratio = 0.0

# Calculate spread
if snapshot.bids and snapshot.asks:
best_bid = max(bid.price for bid in snapshot.bids)
best_ask = min(ask.price for ask in snapshot.asks)
spread = (best_ask - best_bid) / best_bid
else:
spread = 0.0

# Calculate depth score
depth_score = min(len(snapshot.bids) + len(snapshot.asks), 100) / 100.0

# Create mathematical signature
mathematical_signature = self._create_liquidity_signature(
bid_liquidity, ask_liquidity, imbalance_ratio, spread, depth_score
)

return LiquidityAnalysis(
bid_liquidity=bid_liquidity,
ask_liquidity=ask_liquidity,
imbalance_ratio=imbalance_ratio,
spread=spread,
depth_score=depth_score,
mathematical_signature=mathematical_signature
)

except Exception as e:
self.logger.error(f"❌ Error analyzing liquidity: {e}")
return LiquidityAnalysis(
bid_liquidity=0.0,
ask_liquidity=0.0,
imbalance_ratio=0.0,
spread=0.0,
depth_score=0.0
)

def _perform_mathematical_analysis(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
"""Perform mathematical analysis on order book data."""
try:
if not self.math_orchestrator:
return {}

# Prepare data for mathematical analysis
bid_prices = [bid.price for bid in snapshot.bids]
bid_quantities = [bid.quantity for bid in snapshot.bids]
ask_prices = [ask.price for ask in snapshot.asks]
ask_quantities = [ask.quantity for ask in snapshot.asks]

# Combine data for analysis
analysis_data = np.array(bid_prices + bid_quantities + ask_prices + ask_quantities)

# Perform mathematical orchestration
result = self.math_orchestrator.process_data(analysis_data)

return {
'mathematical_score': float(result),
'bid_count': len(snapshot.bids),
'ask_count': len(snapshot.asks),
'total_levels': len(snapshot.bids) + len(snapshot.asks),
'timestamp': time.time()
}

except Exception as e:
self.logger.error(f"❌ Error performing mathematical analysis: {e}")
return {}


def _classify_market_structure(self, snapshot: OrderBookSnapshot, -> None
walls: List[OrderBookWall],
liquidity_analysis: LiquidityAnalysis) -> MarketStructure:
"""Classify market structure based on analysis."""
try:
# Count wall types
buy_walls = sum(1 for wall in walls if wall.wall_type == WallType.BUY_WALL)
sell_walls = sum(1 for wall in walls if wall.wall_type == WallType.SELL_WALL)

# Analyze imbalance
imbalance = abs(liquidity_analysis.imbalance_ratio)

# Classify based on analysis
if imbalance > 0.1:  # Strong imbalance
if liquidity_analysis.imbalance_ratio > 0:
return MarketStructure.BULLISH
else:
return MarketStructure.BEARISH
elif buy_walls > sell_walls and buy_walls > 0:
return MarketStructure.BULLISH
elif sell_walls > buy_walls and sell_walls > 0:
return MarketStructure.BEARISH
elif liquidity_analysis.spread > 0.01:  # High spread
return MarketStructure.VOLATILE
else:
return MarketStructure.NEUTRAL

except Exception as e:
self.logger.error(f"❌ Error classifying market structure: {e}")
return MarketStructure.NEUTRAL

def _calculate_wall_strength(self, quantity: float, levels: int, price: float) -> float:
"""Calculate wall strength using mathematical infrastructure."""
try:
if not self.math_orchestrator:
# Fallback calculation
return min(quantity / 1000.0, 1.0)  # Normalize to 1000 units

# Use mathematical orchestration for wall strength calculation
wall_data = np.array([quantity, levels, price])
strength = self.math_orchestrator.process_data(wall_data)
return float(strength)

except Exception as e:
self.logger.error(f"❌ Error calculating wall strength: {e}")
return 0.0


def _create_wall_signature(self, price: float, quantity: float, -> None
strength: float, wall_type: str) -> str:
"""Create mathematical signature for wall."""
try:
signature_data = np.array([price, quantity, strength])
if self.math_orchestrator:
signature = self.math_orchestrator.process_data(signature_data)
return f"{wall_type}_{signature:.6f}"
else:
return f"{wall_type}_{price:.2f}_{quantity:.2f}_{strength:.3f}"
except Exception as e:
self.logger.error(f"❌ Error creating wall signature: {e}")
return f"{wall_type}_fallback"


def _create_liquidity_signature(self, bid_liquidity: float, ask_liquidity: float, -> None
imbalance: float, spread: float, depth: float) -> str:
"""Create mathematical signature for liquidity analysis."""
try:
liquidity_data = np.array([bid_liquidity, ask_liquidity, imbalance, spread, depth])
if self.math_orchestrator:
signature = self.math_orchestrator.process_data(liquidity_data)
return f"liquidity_{signature:.6f}"
else:
return f"liquidity_{imbalance:.3f}_{spread:.3f}_{depth:.3f}"
except Exception as e:
self.logger.error(f"❌ Error creating liquidity signature: {e}")
return "liquidity_fallback"

def _wall_to_dict(self, wall: OrderBookWall) -> Dict[str, Any]:
"""Convert wall to dictionary."""
return {
'wall_type': wall.wall_type.value,
'price_level': wall.price_level,
'total_quantity': wall.total_quantity,
'strength': wall.strength,
'confidence': wall.confidence,
'mathematical_signature': wall.mathematical_signature,
'timestamp': wall.timestamp
}

def _liquidity_to_dict(self, liquidity: LiquidityAnalysis) -> Dict[str, Any]:
"""Convert liquidity analysis to dictionary."""
return {
'bid_liquidity': liquidity.bid_liquidity,
'ask_liquidity': liquidity.ask_liquidity,
'imbalance_ratio': liquidity.imbalance_ratio,
'spread': liquidity.spread,
'depth_score': liquidity.depth_score,
'mathematical_signature': liquidity.mathematical_signature
}

def get_order_book_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
"""Get latest order book analysis for a symbol."""
if symbol in self.order_book_snapshots:
snapshot = self.order_book_snapshots[symbol]
return {
'symbol': snapshot.symbol,
'timestamp': snapshot.timestamp,
'market_structure': snapshot.market_structure.value,
'bid_count': len(snapshot.bids),
'ask_count': len(snapshot.asks),
'mathematical_analysis': snapshot.mathematical_analysis
}
return None

def get_wall_history(self, symbol: str) -> List[Dict[str, Any]]:
"""Get wall history for a symbol."""
if symbol in self.wall_history:
return [self._wall_to_dict(wall) for wall in self.wall_history[symbol][-50:]]
return []

def get_liquidity_history(self, symbol: str) -> List[Dict[str, Any]]:
"""Get liquidity history for a symbol."""
if symbol in self.liquidity_history:
return [self._liquidity_to_dict(liquidity) for liquidity in self.liquidity_history[symbol][-50:]]
return []

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get system performance metrics."""
return {
'snapshots_analyzed': self.metrics.snapshots_analyzed,
'walls_detected': self.metrics.walls_detected,
'liquidity_analyses': self.metrics.liquidity_analyses,
'mathematical_analyses': self.metrics.mathematical_analyses,
'average_processing_time': self.metrics.average_processing_time,
'last_updated': time.time()
}

def activate(self) -> bool:
"""Activate the system."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True
self.logger.info("✅ Order Book Analyzer System activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating Order Book Analyzer System: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the system."""
try:
self.active = False
self.logger.info("✅ Order Book Analyzer System deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating Order Book Analyzer System: {e}")
return False

def get_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
'active': self.active,
'initialized': self.initialized,
'symbols_tracked': len(self.order_book_snapshots),
'performance_metrics': self.get_performance_metrics(),
'config': {
'enabled': self.config.enabled,
'wall_detection_threshold': self.config.wall_detection_threshold,
'imbalance_threshold': self.config.imbalance_threshold,
'max_depth_levels': self.config.max_depth_levels,
'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled
}
}

def calculate_mathematical_result(self, data: Union[List, np.ndarray]) -> float:
"""Calculate mathematical result with proper data handling and order book analysis integration."""
try:
if not isinstance(data, np.ndarray):
data = np.array(data)

if MATH_INFRASTRUCTURE_AVAILABLE and self.math_orchestrator:
# Use the actual mathematical modules for calculation
if len(data) > 0:
# Use mathematical orchestration for order book analysis
result = self.math_orchestrator.process_data(data)
return float(result)
else:
return 0.0
else:
# Fallback to basic calculation
result = np.sum(data) / len(data) if len(data) > 0 else 0.0
return float(result)
except Exception as e:
self.logger.error(f"Mathematical calculation error: {e}")
return 0.0

def create_order_book_analyzer(config: Optional[OrderBookAnalyzerConfig] = None) -> OrderBookAnalyzer:
"""Factory function to create OrderBookAnalyzer instance."""
return OrderBookAnalyzer(config)

def main():
"""Main function for testing."""
# Create configuration
config = OrderBookAnalyzerConfig(
enabled=True,
debug=True,
wall_detection_threshold=0.1,
imbalance_threshold=0.05,
max_depth_levels=20,
mathematical_analysis_enabled=True
)

# Create analyzer
analyzer = create_order_book_analyzer(config)

# Activate system
analyzer.activate()

# Test order book data
test_order_book = {
'bids': [
[50000.0, 1.5],
[49999.0, 2.0],
[49998.0, 1.8],
[49997.0, 3.2],
[49996.0, 1.2]
],
'asks': [
[50001.0, 1.0],
[50002.0, 2.5],
[50003.0, 1.8],
[50004.0, 2.1],
[50005.0, 1.5]
]
}

# Analyze order book
result = analyzer.analyze_order_book("BTCUSDT", test_order_book)
print(f"Analysis Result: {json.dumps(result, indent=2)}")

# Get status
status = analyzer.get_status()
print(f"System Status: {json.dumps(status, indent=2)}")

# Deactivate system
analyzer.deactivate()

if __name__ == "__main__":
main()
