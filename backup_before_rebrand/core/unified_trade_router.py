"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Trade Router Module
============================
Provides unified trade routing functionality for the Schwabot trading system.

Mathematical Core:
R(t) = {
Route to Market Maker,    if L(t) > θ_l
Route to Taker,          if L(t) < θ_t
Route to Hybrid,         else
}
Where:
- L(t): liquidity score at time t
- θ_l, θ_t: liquidity thresholds

This module coordinates all execution components and routes trades to optimal
execution venues based on mathematical analysis and market conditions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

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


class RoutingVenue(Enum):
"""Class for Schwabot trading functionality."""
"""Routing venue types."""
MARKET_MAKER = "market_maker"
TAKER = "taker"
HYBRID = "hybrid"
SMART_ROUTER = "smart_router"
DARK_POOL = "dark_pool"
AGGREGATOR = "aggregator"


class LiquidityLevel(Enum):
"""Class for Schwabot trading functionality."""
"""Liquidity level classifications."""
HIGH = "high"
MEDIUM = "medium"
LOW = "low"
CRITICAL = "critical"


class RoutingPriority(Enum):
"""Class for Schwabot trading functionality."""
"""Routing priority levels."""
ULTRA_HIGH = "ultra_high"
HIGH = "high"
NORMAL = "normal"
LOW = "low"
BULK = "bulk"


@dataclass
class TradeRequest:
"""Class for Schwabot trading functionality."""
"""Trade request with mathematical properties."""
request_id: str
symbol: str
side: str  # 'buy' or 'sell'
quantity: float
price: Optional[float] = None
urgency: RoutingPriority = RoutingPriority.NORMAL
timestamp: float = field(default_factory=time.time)
mathematical_signature: str = ""
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
"""Class for Schwabot trading functionality."""
"""Routing decision with mathematical analysis."""
request_id: str
selected_venue: RoutingVenue
routing_score: float
confidence: float
reasoning: str
mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
timestamp: float = field(default_factory=time.time)
routing_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenuePerformance:
"""Class for Schwabot trading functionality."""
"""Venue performance metrics."""
venue: RoutingVenue
total_routes: int
successful_routes: int
average_latency: float
average_slippage: float
success_rate: float
mathematical_signature: str = ""


@dataclass
class LiquidityData:
"""Class for Schwabot trading functionality."""
"""Liquidity data for routing decisions."""
symbol: str
liquidity_level: LiquidityLevel
liquidity_score: float
spread: float
depth: float
mathematical_signature: str = ""


@dataclass
class UnifiedTradeRouterConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for unified trade router."""
enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False
max_concurrent_routes: int = 50
routing_threshold: float = 0.7  # Minimum routing score
mathematical_analysis_enabled: bool = True
adaptive_routing_enabled: bool = True
liquidity_thresholds: Dict[str, float] = field(default_factory=lambda: {
'high_liquidity': 0.8,
'low_liquidity': 0.3
})
venue_weights: Dict[str, float] = field(default_factory=lambda: {
'market_maker': 0.4,
'taker': 0.3,
'hybrid': 0.2,
'smart_router': 0.1
})


class UnifiedTradeRouter:
"""Class for Schwabot trading functionality."""
"""
Unified Trade Router System

Implements intelligent trade routing:
R(t) = {
Route to Market Maker,    if L(t) > θ_l
Route to Taker,          if L(t) < θ_t
Route to Hybrid,         else
}

Coordinates all execution components and routes trades to optimal
execution venues based on mathematical analysis and market conditions.
"""

def __init__(self, config: Optional[UnifiedTradeRouterConfig] = None) -> None:
"""Initialize the unified trade router system."""
self.config = config or UnifiedTradeRouterConfig()
self.logger = logging.getLogger(__name__)

# Routing state
self.active_routes: Dict[str, RoutingDecision] = {}
self.routing_history: List[RoutingDecision] = []
self.venue_performance: Dict[RoutingVenue, VenuePerformance] = {}
self.liquidity_data: Dict[str, LiquidityData] = {}

# Request processing
self.request_queue: asyncio.Queue = asyncio.Queue()
self.routing_queue: asyncio.Queue = asyncio.Queue()

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
'requests_processed': 0,
'routes_executed': 0,
'routing_errors': 0,
'average_routing_time': 0.0,
'routing_accuracy': 0.0
}

# System state
self.initialized = False
self.active = False

self._initialize_system()

def _initialize_system(self) -> None:
"""Initialize the unified trade router system."""
try:
self.logger.info("Initializing Unified Trade Router System")

# Initialize venue performance tracking
for venue in RoutingVenue:
self.venue_performance[venue] = VenuePerformance(
venue=venue,
total_routes=0,
successful_routes=0,
average_latency=0.0,
average_slippage=0.0,
success_rate=0.0
)

self.initialized = True
self.logger.info("✅ Unified Trade Router System initialized successfully")

except Exception as e:
self.logger.error(f"❌ Error initializing Unified Trade Router System: {e}")
self.initialized = False

async def start_router(self) -> bool:
"""Start the unified trade router."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True

# Start processing tasks
asyncio.create_task(self._process_request_queue())
asyncio.create_task(self._process_routing_queue())

self.logger.info("✅ Unified Trade Router started")
return True

except Exception as e:
self.logger.error(f"❌ Error starting unified trade router: {e}")
return False

async def stop_router(self) -> bool:
"""Stop the unified trade router."""
try:
self.active = False
self.logger.info("✅ Unified Trade Router stopped")
return True

except Exception as e:
self.logger.error(f"❌ Error stopping unified trade router: {e}")
return False

async def route_trade(self, trade_data: Dict[str, Any]) -> bool:
"""Route a trade request to optimal venue."""
if not self.active:
self.logger.error("Trade router not active")
return False

try:
# Validate trade data
if not self._validate_trade_data(trade_data):
self.logger.error(f"Invalid trade data: {trade_data}")
return False

# Create trade request
request = self._create_trade_request(trade_data)

# Add mathematical analysis
if self.config.mathematical_analysis_enabled:
await self._analyze_trade_mathematically(request)

# Queue for processing
await self.request_queue.put(request)

self.logger.info(f"✅ Trade request queued: {request.request_id} for {request.symbol}")
return True

except Exception as e:
self.logger.error(f"❌ Error routing trade: {e}")
return False

def _validate_trade_data(self, trade_data: Dict[str, Any]) -> bool:
"""Validate trade data."""
try:
required_fields = ['symbol', 'side', 'quantity']

for field in required_fields:
if field not in trade_data:
return False

# Check quantity
quantity = trade_data.get('quantity', 0)
if quantity <= 0:
return False

# Check side
side = trade_data.get('side', '').lower()
if side not in ['buy', 'sell']:
return False

return True

except Exception as e:
self.logger.error(f"❌ Error validating trade data: {e}")
return False

def _create_trade_request(self, trade_data: Dict[str, Any]) -> TradeRequest:
"""Create a trade request from trade data."""
try:
# Generate request ID
request_id = f"TRADE_{trade_data.get('symbol', '')}_{int(time.time() * 1000)}"

# Determine urgency
urgency = RoutingPriority(trade_data.get('urgency', 'normal'))

# Create request
request = TradeRequest(
request_id=request_id,
symbol=trade_data.get('symbol'),
side=trade_data.get('side'),
quantity=float(trade_data.get('quantity')),
price=float(trade_data.get('price')) if trade_data.get('price') else None,
urgency=urgency,
metadata=trade_data.get('metadata', {})
)

return request

except Exception as e:
self.logger.error(f"❌ Error creating trade request: {e}")
raise

async def _analyze_trade_mathematically(self, request: TradeRequest) -> None:
"""Perform mathematical analysis on trade request."""
try:
if not self.math_bridge:
return

# Prepare trade data for mathematical analysis
trade_data = {
'request_id': request.request_id,
'symbol': request.symbol,
'side': request.side,
'quantity': request.quantity,
'price': request.price,
'urgency': request.urgency.value,
'timestamp': request.timestamp,
'metadata': request.metadata
}

# Perform mathematical integration
result = self.math_bridge.integrate_all_mathematical_systems(
trade_data, {}
)

# Update request with mathematical analysis
request.mathematical_signature = result.mathematical_signature
request.metadata['mathematical_analysis'] = {
'confidence': result.overall_confidence,
'connections': len(result.connections),
'performance_metrics': result.performance_metrics
}

except Exception as e:
self.logger.error(f"❌ Error analyzing trade mathematically: {e}")

async def _process_request_queue(self) -> None:
"""Process trade requests from the queue."""
try:
while self.active:
try:
# Get request from queue
request = await asyncio.wait_for(
self.request_queue.get(),
timeout=1.0
)

# Process request
await self._process_trade_request(request)

# Mark task as done
self.request_queue.task_done()

except asyncio.TimeoutError:
continue
except Exception as e:
self.logger.error(f"❌ Error processing trade request: {e}")

except Exception as e:
self.logger.error(f"❌ Error in request processing loop: {e}")

async def _process_trade_request(self, request: TradeRequest) -> None:
"""Process a trade request."""
try:
start_time = time.time()

# Update performance metrics
self.performance_metrics['requests_processed'] += 1

# Get liquidity data
liquidity_data = self._get_liquidity_data(request.symbol)

# Make routing decision
decision = await self._make_routing_decision(request, liquidity_data)

# Store decision
self.routing_history.append(decision)

# Update venue performance
self._update_venue_performance(decision)

# Queue for execution
await self.routing_queue.put(decision)

# Update performance metrics
routing_time = time.time() - start_time
self.performance_metrics['average_routing_time'] = (
(self.performance_metrics['average_routing_time'] * (self.performance_metrics['requests_processed'] - 1) + routing_time) /
self.performance_metrics['requests_processed']
)

self.logger.info(f"✅ Trade request processed: {request.request_id} -> {decision.selected_venue.value}")

except Exception as e:
self.logger.error(f"❌ Error processing trade request: {e}")
self.performance_metrics['routing_errors'] += 1

def _get_liquidity_data(self, symbol: str) -> LiquidityData:
"""Get liquidity data for symbol."""
try:
# Use cached data or create default
if symbol in self.liquidity_data:
return self.liquidity_data[symbol]

# Create default liquidity data
default_data = LiquidityData(
symbol=symbol,
liquidity_level=LiquidityLevel.MEDIUM,
liquidity_score=0.5,
spread=0.001,  # 0.1%
depth=1000.0
)

self.liquidity_data[symbol] = default_data
return default_data

except Exception as e:
self.logger.error(f"❌ Error getting liquidity data: {e}")
return LiquidityData(
symbol=symbol,
liquidity_level=LiquidityLevel.MEDIUM,
liquidity_score=0.5,
spread=0.001,
depth=1000.0
)

async def _make_routing_decision(self, request: TradeRequest, liquidity_data: LiquidityData) -> RoutingDecision:
"""Make routing decision based on mathematical optimization."""
try:
# Calculate routing scores for each venue
routing_scores = {}
for venue in RoutingVenue:
score = await self._calculate_venue_score(request, venue, liquidity_data)
routing_scores[venue] = score

# Select optimal venue
selected_venue = max(routing_scores.items(), key=lambda x: x[1])[0]
routing_score = routing_scores[selected_venue]

# Determine if routing should proceed
should_route = routing_score >= self.config.routing_threshold

if not should_route:
selected_venue = RoutingVenue.SMART_ROUTER  # Default fallback
routing_score = 0.0

# Generate routing parameters
routing_parameters = self._generate_routing_parameters(request, selected_venue, liquidity_data)

# Perform mathematical analysis on decision
mathematical_analysis = await self._analyze_decision_mathematically(
request, selected_venue, routing_score, liquidity_data
)

# Create reasoning
reasoning = self._generate_routing_reasoning(
request, selected_venue, routing_score, liquidity_data
)

return RoutingDecision(
request_id=request.request_id,
selected_venue=selected_venue,
routing_score=routing_score,
confidence=request.metadata.get('mathematical_analysis', {}).get('confidence', 0.5),
reasoning=reasoning,
mathematical_analysis=mathematical_analysis,
routing_parameters=routing_parameters
)

except Exception as e:
self.logger.error(f"❌ Error making routing decision: {e}")
return RoutingDecision(
request_id=request.request_id,
selected_venue=RoutingVenue.SMART_ROUTER,
routing_score=0.0,
confidence=0.0,
reasoning=f"Error in routing decision: {e}",
routing_parameters={}
)

async def _calculate_venue_score(self, request: TradeRequest, venue: RoutingVenue, liquidity_data: LiquidityData) -> float:
"""Calculate routing score for a venue using mathematical optimization."""
try:
# Get venue weight
venue_weight = self.config.venue_weights.get(venue.value, 0.1)

# Get venue performance
performance = self.venue_performance[venue]

# Calculate base score
base_score = performance.success_rate * venue_weight

# Adjust for liquidity conditions
liquidity_adjustment = self._calculate_liquidity_adjustment(request, venue, liquidity_data)

# Adjust for urgency
urgency_adjustment = self._calculate_urgency_adjustment(request, venue)

# Apply mathematical optimization
routing_score = (base_score * 0.4 +
liquidity_adjustment * 0.4 +
urgency_adjustment * 0.2)

return max(0.0, min(1.0, routing_score))

except Exception as e:
self.logger.error(f"❌ Error calculating venue score: {e}")
return 0.0

def _calculate_liquidity_adjustment(self, request: TradeRequest, venue: RoutingVenue, liquidity_data: LiquidityData) -> float:
"""Calculate liquidity adjustment for venue."""
try:
liquidity_score = liquidity_data.liquidity_score

# Venue-specific liquidity preferences
if venue == RoutingVenue.MARKET_MAKER:
# Market makers prefer high liquidity
if liquidity_score > self.config.liquidity_thresholds['high_liquidity']:
return 0.3
elif liquidity_score < self.config.liquidity_thresholds['low_liquidity']:
return -0.3
else:
return 0.0

elif venue == RoutingVenue.TAKER:
# Takers can handle lower liquidity
if liquidity_score < self.config.liquidity_thresholds['low_liquidity']:
return 0.2
else:
return 0.0

elif venue == RoutingVenue.HYBRID:
# Hybrid adapts to liquidity
return 0.0

elif venue == RoutingVenue.SMART_ROUTER:
# Smart router optimizes based on conditions
if liquidity_score > 0.7:
return 0.1
elif liquidity_score < 0.3:
return 0.2
else:
return 0.0

return 0.0

except Exception as e:
self.logger.error(f"❌ Error calculating liquidity adjustment: {e}")
return 0.0

def _calculate_urgency_adjustment(self, request: TradeRequest, venue: RoutingVenue) -> float:
"""Calculate urgency adjustment for venue."""
try:
urgency = request.urgency

# Venue-specific urgency preferences
if venue == RoutingVenue.MARKET_MAKER:
# Market makers prefer normal urgency
if urgency == RoutingPriority.NORMAL:
return 0.2
elif urgency == RoutingPriority.ULTRA_HIGH:
return -0.2
else:
return 0.0

elif venue == RoutingVenue.TAKER:
# Takers handle high urgency well
if urgency in [RoutingPriority.ULTRA_HIGH, RoutingPriority.HIGH]:
return 0.3
elif urgency == RoutingPriority.BULK:
return -0.2
else:
return 0.0

elif venue == RoutingVenue.HYBRID:
# Hybrid adapts to urgency
if urgency == RoutingPriority.NORMAL:
return 0.1
else:
return 0.0

elif venue == RoutingVenue.SMART_ROUTER:
# Smart router optimizes for all urgencies
return 0.1

return 0.0

except Exception as e:
self.logger.error(f"❌ Error calculating urgency adjustment: {e}")
return 0.0

def _generate_routing_parameters(self, request: TradeRequest, selected_venue: RoutingVenue, liquidity_data: LiquidityData) -> Dict[str, Any]:
"""Generate routing parameters."""
try:
base_params = {
'venue': selected_venue.value,
'symbol': request.symbol,
'side': request.side,
'quantity': request.quantity,
'urgency': request.urgency.value,
'liquidity_level': liquidity_data.liquidity_level.value,
'liquidity_score': liquidity_data.liquidity_score,
'spread': liquidity_data.spread,
'depth': liquidity_data.depth
}

# Add venue-specific parameters
if selected_venue == RoutingVenue.MARKET_MAKER:
base_params.update({
'order_type': 'limit',
'price_offset': 0.0005,  # 0.05% offset
'timeout': 300  # 5 minutes
})
elif selected_venue == RoutingVenue.TAKER:
base_params.update({
'order_type': 'market',
'slippage_tolerance': 0.002,  # 0.2%
'timeout': 30  # 30 seconds
})
elif selected_venue == RoutingVenue.HYBRID:
base_params.update({
'order_type': 'smart',
'adaptive_pricing': True,
'timeout': 120  # 2 minutes
})
elif selected_venue == RoutingVenue.SMART_ROUTER:
base_params.update({
'order_type': 'optimal',
'multi_venue': True,
'timeout': 60  # 1 minute
})

return base_params

except Exception as e:
self.logger.error(f"❌ Error generating routing parameters: {e}")
return {'error': str(e)}

async def _analyze_decision_mathematically(self, request: TradeRequest, selected_venue: RoutingVenue,
routing_score: float, liquidity_data: LiquidityData) -> Dict[str, Any]:
"""Perform mathematical analysis on routing decision."""
try:
if not self.math_bridge:
return {}

# Prepare decision data for mathematical analysis
decision_data = {
'request_id': request.request_id,
'selected_venue': selected_venue.value,
'routing_score': routing_score,
'liquidity_level': liquidity_data.liquidity_level.value,
'liquidity_score': liquidity_data.liquidity_score,
'urgency': request.urgency.value,
'mathematical_signature': request.mathematical_signature
}

# Perform mathematical integration
result = self.math_bridge.integrate_all_mathematical_systems(
decision_data, {}
)

return {
'confidence': result.overall_confidence,
'connections': len(result.connections),
'performance_metrics': result.performance_metrics,
'mathematical_signature': result.mathematical_signature
}

except Exception as e:
self.logger.error(f"❌ Error analyzing decision mathematically: {e}")
return {}

def _generate_routing_reasoning(self, request: TradeRequest, selected_venue: RoutingVenue, -> None
routing_score: float, liquidity_data: LiquidityData) -> str:
"""Generate human-readable routing reasoning."""
try:
reasoning_parts = []

# Venue selection reason
reasoning_parts.append(f"Selected {selected_venue.value} venue with routing score {routing_score:.3f}")

# Liquidity context
reasoning_parts.append(f"Liquidity: {liquidity_data.liquidity_level.value} (score: {liquidity_data.liquidity_score:.3f})")

# Urgency context
reasoning_parts.append(f"Urgency: {request.urgency.value}")

# Market conditions
reasoning_parts.append(f"Spread: {liquidity_data.spread:.4f}, Depth: {liquidity_data.depth:.0f}")

return " | ".join(reasoning_parts)

except Exception as e:
self.logger.error(f"❌ Error generating routing reasoning: {e}")
return f"Error generating reasoning: {e}"

def _update_venue_performance(self, decision: RoutingDecision) -> None:
"""Update venue performance metrics."""
try:
venue = decision.selected_venue
performance = self.venue_performance[venue]

# Update metrics
performance.total_routes += 1
performance.success_rate = performance.successful_routes / performance.total_routes if performance.total_routes > 0 else 0.0

# Update mathematical signature
performance.mathematical_signature = decision.mathematical_analysis.get('mathematical_signature', '')

# Note: successful_routes, average_latency, and average_slippage would be updated after execution results

except Exception as e:
self.logger.error(f"❌ Error updating venue performance: {e}")

async def _process_routing_queue(self) -> None:
"""Process routing decisions from the queue."""
try:
while self.active:
try:
# Get decision from queue
decision = await asyncio.wait_for(
self.routing_queue.get(),
timeout=1.0
)

# Process decision (send to execution engine)
await self._execute_routing_decision(decision)

# Mark task as done
self.routing_queue.task_done()

except asyncio.TimeoutError:
continue
except Exception as e:
self.logger.error(f"❌ Error processing routing decision: {e}")

except Exception as e:
self.logger.error(f"❌ Error in routing processing loop: {e}")

async def _execute_routing_decision(self, decision: RoutingDecision) -> None:
"""Execute a routing decision (send to execution engine)."""
try:
# Update performance metrics
self.performance_metrics['routes_executed'] += 1

# Log execution
self.logger.info(f"🚀 Executing routing decision: {decision.request_id} -> {decision.selected_venue.value}")

# Here you would send the decision to the appropriate execution engine
# For now, we'll just log it
execution_data = {
'decision_id': decision.request_id,
'selected_venue': decision.selected_venue.value,
'routing_score': decision.routing_score,
'confidence': decision.confidence,
'parameters': decision.routing_parameters,
'timestamp': decision.timestamp
}

self.logger.info(f"Routing execution data: {json.dumps(execution_data, indent=2)}")

except Exception as e:
self.logger.error(f"❌ Error executing routing decision: {e}")

def update_liquidity_data(self, symbol: str, liquidity_data: Dict[str, Any]) -> bool:
"""Update liquidity data for a symbol."""
try:
liquidity_level = LiquidityLevel(liquidity_data.get('level', 'medium'))

data = LiquidityData(
symbol=symbol,
liquidity_level=liquidity_level,
liquidity_score=liquidity_data.get('score', 0.5),
spread=liquidity_data.get('spread', 0.001),
depth=liquidity_data.get('depth', 1000.0),
mathematical_signature=liquidity_data.get('mathematical_signature', '')
)

self.liquidity_data[symbol] = data

self.logger.info(f"✅ Updated liquidity data for {symbol}: {liquidity_level.value}")
return True

except Exception as e:
self.logger.error(f"❌ Error updating liquidity data: {e}")
return False

def get_venue_performance(self, venue: Optional[RoutingVenue] = None) -> Dict[str, Any]:
"""Get venue performance metrics."""
try:
if venue:
performance = self.venue_performance[venue]
return {
'venue': performance.venue.value,
'total_routes': performance.total_routes,
'successful_routes': performance.successful_routes,
'average_latency': performance.average_latency,
'average_slippage': performance.average_slippage,
'success_rate': performance.success_rate,
'mathematical_signature': performance.mathematical_signature
}
else:
return {
venue.value: {
'total_routes': perf.total_routes,
'successful_routes': perf.successful_routes,
'average_latency': perf.average_latency,
'average_slippage': perf.average_slippage,
'success_rate': perf.success_rate
}
for venue, perf in self.venue_performance.items()
}

except Exception as e:
self.logger.error(f"❌ Error getting venue performance: {e}")
return {}

def get_recent_routing_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
"""Get recent routing decisions."""
try:
recent_decisions = self.routing_history[-limit:]
return [
{
'request_id': decision.request_id,
'selected_venue': decision.selected_venue.value,
'routing_score': decision.routing_score,
'confidence': decision.confidence,
'reasoning': decision.reasoning,
'timestamp': decision.timestamp,
'routing_parameters': decision.routing_parameters
}
for decision in recent_decisions
]
except Exception as e:
self.logger.error(f"❌ Error getting recent routing decisions: {e}")
return []

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get system performance metrics."""
metrics = self.performance_metrics.copy()

# Calculate routing accuracy
total_routes = metrics['routes_executed']
if total_routes > 0:
metrics['routing_accuracy'] = metrics['routes_executed'] / total_routes
else:
metrics['routing_accuracy'] = 0.0

return metrics

def activate(self) -> bool:
"""Activate the system."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True
self.logger.info("✅ Unified Trade Router System activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating Unified Trade Router System: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the system."""
try:
self.active = False
self.logger.info("✅ Unified Trade Router System deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating Unified Trade Router System: {e}")
return False

def get_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
'active': self.active,
'initialized': self.initialized,
'requests_queued': self.request_queue.qsize(),
'routing_queued': self.routing_queue.qsize(),
'active_routes': len(self.active_routes),
'total_routing_decisions': len(self.routing_history),
'symbols_tracked': len(self.liquidity_data),
'performance_metrics': self.performance_metrics,
'config': {
'enabled': self.config.enabled,
'max_concurrent_routes': self.config.max_concurrent_routes,
'routing_threshold': self.config.routing_threshold,
'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled,
'adaptive_routing_enabled': self.config.adaptive_routing_enabled
}
}


def create_unified_trade_router(config: Optional[UnifiedTradeRouterConfig] = None) -> UnifiedTradeRouter:
"""Factory function to create UnifiedTradeRouter instance."""
return UnifiedTradeRouter(config)


async def main():
"""Main function for testing."""
# Create configuration
config = UnifiedTradeRouterConfig(
enabled=True,
debug=True,
max_concurrent_routes=20,
routing_threshold=0.7,
mathematical_analysis_enabled=True,
adaptive_routing_enabled=True
)

# Create router
router = create_unified_trade_router(config)

# Activate system
router.activate()

# Start router
await router.start_router()

# Update liquidity data
router.update_liquidity_data("BTCUSDT", {
'level': 'high',
'score': 0.85,
'spread': 0.0005,
'depth': 5000.0
})

# Submit test trades
test_trades = [
{
'symbol': 'BTCUSDT',
'side': 'buy',
'quantity': 0.1,
'urgency': 'normal',
'metadata': {'test': True}
},
{
'symbol': 'ETHUSDT',
'side': 'sell',
'quantity': 1.0,
'urgency': 'high',
'metadata': {'test': True}
},
{
'symbol': 'BTCUSDT',
'side': 'buy',
'quantity': 0.5,
'urgency': 'ultra_high',
'metadata': {'test': True}
}
]

# Route trades
for trade_data in test_trades:
await router.route_trade(trade_data)

# Wait for processing
await asyncio.sleep(5)

# Get status
status = router.get_status()
print(f"System Status: {json.dumps(status, indent=2)}")

# Get venue performance
performance = router.get_venue_performance()
print(f"Venue Performance: {json.dumps(performance, indent=2)}")

# Get recent routing decisions
decisions = router.get_recent_routing_decisions()
print(f"Recent Routing Decisions: {json.dumps(decisions, indent=2)}")

# Stop router
await router.stop_router()

# Deactivate system
router.deactivate()


if __name__ == "__main__":
asyncio.run(main())
