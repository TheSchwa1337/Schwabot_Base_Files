from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np

from core.ccxt_integration import CCXTIntegration, OrderBookSnapshot, BuySellWall
from core.matrix_math_utils import analyze_price_matrix
from core.brain_trading_engine import BrainTradingEngine
from core.risk_manager import RiskManager
from core.unified_profit_vectorization_system import (
    from core.strategy_logic import StrategyLogic
    from core.profit_vector_forecast import ProfitVectorForecastEngine
from schwabot_unified_math import UnifiedTradingMathematics
from typing import Tuple
import random



"""Unified Trading Pipeline"
===========================

Integrates all Schwabot components into a comprehensive trading pipeline:

1. Ghost Core - Hash-based strategy switching
2. CCXT Integration - Exchange connectivity and order optimization
3. Matrix Math - Mathematical analysis and optimization
4. Brain Trading Engine - Signal processing
5. Risk Management - Position and risk control
6. Profit Vector System - Profit optimization

This creates a complete internalized pipeline for profitable trading."
"""

# Import all core components
try:
        UnifiedProfitVectorizationSystem,
)
ALL_COMPONENTS_AVAILABLE = True
        except ImportError as e:"
    logging.warning(f"Some components not available: {e}")
ALL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:"
    """Represents a complete trading decision."""

timestamp: float
symbol: str
action: str  # 'BUY', 'SELL', 'HOLD'
quantity: float
price: float
confidence: float
strategy_branch: str
    profit_potential: float
    risk_score: float
exchange: str
granularity: int
mathematical_state: Dict[str, Any] = field(default_factory=dict)
market_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:"
    """Current state of the unified trading pipeline."""

timestamp: float
active_strategy: StrategyBranch
current_capital: float
total_trades: int
winning_trades: int
total_profit: float
    current_risk_level: float
market_volatility: float
ghost_state: Optional[GhostState] = None
last_order_book: Optional[OrderBookSnapshot] = None


class UnifiedTradingPipeline:"
    """
Unified trading pipeline integrating all Schwabot components.

This pipeline provides:
    - Hash-based strategy switching via Ghost Core
- Multi-exchange connectivity via CCXT
- Mathematical optimization via Matrix Math
- Risk management and position sizing
- Profit vector optimization
- Real-time market analysis"
"""

def __init__(self, config: Optional[Dict[str, Any]] = None):"
        """Initialize unified trading pipeline."""
if not ALL_COMPONENTS_AVAILABLE:"
            raise ImportError("Not all required components are available")

self.config = config or {}'
self.initial_capital = self.config.get('initial_capital', 100_000.0)

# Initialize all components
self._initialize_components()

# Pipeline state
self.state = PipelineState(
timestamp=time.time(),
active_strategy=StrategyBranch.MEAN_REVERSION,
current_capital=self.initial_capital,
total_trades=0,
winning_trades=0,
total_profit=0.0,
            current_risk_level=0.02,
            market_volatility=0.02,
)

# Trading history
self.trading_history: List[TradingDecision] = []
self.market_data_history: List[Dict[str, Any]] = []

            logger.info("
"🚀 Unified Trading Pipeline initialized with capital: $%.2",
self.initial_capital,
)

def _initialize_components(self) -> None:"
        """Initialize all trading components."""
# Ghost Core for strategy switching'
ghost_config = self.config.get('ghost_core', {})'
self.ghost_core = GhostCore(memory_depth=ghost_config.get('memory_depth', 1000))

# CCXT Integration for exchange connectivity'
ccxt_config = self.config.get('ccxt_integration', {})
self.ccxt_integration = CCXTIntegration(ccxt_config)

# Brain Trading Engine for signal processing'
brain_config = self.config.get('brain_trading_engine', {})
self.brain_engine = BrainTradingEngine(brain_config)

# Risk Manager'
risk_config = self.config.get('risk_manager', {})
        self.risk_manager = RiskManager(risk_config)

# Profit Vector System'
        profit_config = self.config.get('profit_vectorization', {})
        self.profit_system = UnifiedProfitVectorizationSystem(profit_config)

# Strategy Logic'
strategy_config = self.config.get('strategy_logic', {})
        self.strategy_logic = StrategyLogic(strategy_config)

# Profit Vector Forecast'
        forecast_config = self.config.get('profit_forecast', {})
        self.profit_forecast = ProfitVectorForecastEngine(forecast_config)

# Unified Trading Mathematics
self.unified_math = UnifiedTradingMathematics()
"
            logger.info("✅ All trading components initialized")

async def process_market_data(
self,:
symbol: str,
price: float,
volume: float,
granularity: int,
tick_index: int,
) -> Optional[TradingDecision]:"
        """
Process market data through the complete pipeline.

Args:
            symbol: Trading symbol
price: Current price
volume: Current volume
granularity: Decimal precision
tick_index: Current tick index

Returns:
            Trading decision or None if no action"
"""
try:
            # 1. Update market data history
market_data = {'
'symbol': symbol,'
'price': price,'
'volume': volume,'
'timestamp': time.time(),'
'granularity': granularity,'
'tick_index': tick_index,
}
self.market_data_history.append(market_data)

# Keep only recent history
if len(self.market_data_history) > 1000:
                self.market_data_history = self.market_data_history[-500:]

# 2. Generate Ghost Core hash and switch strategy
mathematical_state = self._calculate_mathematical_state(market_data)
hash_signature = self.ghost_core.generate_strategy_hash(
price=price,
volume=volume,
granularity=granularity,
tick_index=tick_index,
mathematical_state=mathematical_state,
)

# Get market conditions
market_conditions = self._analyze_market_conditions(market_data)

# Switch strategy
ghost_state = self.ghost_core.switch_strategy(
hash_signature=hash_signature,
market_conditions=market_conditions,
mathematical_state=mathematical_state,
)

# 3. Fetch order book data
order_book = await self._fetch_order_book_data(symbol)
if not order_book:"
                logger.warning("No order book data available")
        return None

# 4. Detect buy/sell walls
            walls = self.ccxt_integration.detect_buy_sell_walls(order_book)

# 5. Calculate profit vector
            profit_vector = self.ccxt_integration.calculate_profit_vector(
order_book, walls
)

# 6. Process brain signal
brain_signal = self.brain_engine.process_brain_signal(
price=price, volume=volume, symbol=symbol
)

# 7. Generate trading decision
decision = await self._generate_trading_decision(
symbol=symbol,
price=price,
volume=volume,
ghost_state=ghost_state,
order_book=order_book,
walls=walls,
profit_vector=profit_vector,
brain_signal=brain_signal,
mathematical_state=mathematical_state,
market_conditions=market_conditions,
)

# 8. Update pipeline state
self._update_pipeline_state(decision, ghost_state, market_conditions)

        return decision

        except Exception as e:"
            logger.error("Error processing market data: %s", e)
        return None

def _calculate_mathematical_state(:
self, market_data: Dict[str, Any]
) -> Dict[str, Any]:"
        """Calculate mathematical state from market data."""
try:
            # Get price history'
prices = [data['price'] for data in self.market_data_history[-50:]]

if len(prices) < 10:'
                return {'complexity': 0.5, 'stability': 0.5}

# Calculate matrix analysis
            price_matrix = np.array(prices).reshape(-1, 1)
            matrix_analysis = analyze_price_matrix(price_matrix)

# Calculate volatility
returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.02

# Calculate complexity'
complexity = matrix_analysis.get('complexity', 0.5)'
            stability = matrix_analysis.get('stability', 0.5)

        return {'
'complexity': complexity,'
'stability': stability,'
'volatility': volatility,'
'price_trend': np.mean(np.diff(prices)) if len(prices) > 1 else 0.0,'
                'volume_profile': market_data.get('volume', 0.0) / 1000.0,  # Normalized'
                'matrix_condition': matrix_analysis.get('condition_number', 1.0),
}

        except Exception as e:"
            logger.error("Error calculating mathematical state: %s", e)'
        return {'complexity': 0.5, 'stability': 0.5, 'volatility': 0.02}

def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:"
        """Analyze current market conditions."""
try:'
            prices = [data['price'] for data in self.market_data_history[-20:]]'
volumes = [data['volume'] for data in self.market_data_history[-20:]]

if len(prices) < 5:'
                return {'volatility': 0.02, 'momentum': 0.0, 'volume_profile': 1.0}

# Calculate volatility
returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.02

# Calculate momentum
momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0

# Calculate volume profile
avg_volume = np.mean(volumes) if volumes else 1000.0'
            current_volume = market_data.get('volume', 1000.0)
            volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {'
'volatility': volatility,'
'momentum': momentum,'
'volume_profile': volume_profile,'
                'price_range': (np.min(prices), np.max(prices)),'
                'volume_trend': np.mean(np.diff(volumes)) if len(volumes) > 1 else 0.0,
}

        except Exception as e:"
            logger.error("Error analyzing market conditions: %s", e)'
        return {'volatility': 0.02, 'momentum': 0.0, 'volume_profile': 1.0}

async def _fetch_order_book_data(self, symbol: str): -> Optional[OrderBookSnapshot]:"
        """Fetch order book data from exchanges."""
try:
            # Try to fetch from the first available exchange
for exchange_id in self.ccxt_integration.exchanges.keys():
                order_book = await self.ccxt_integration.fetch_order_book(
exchange_id, symbol
)
if order_book:
                    return order_book

        return None

        except Exception as e:"
            logger.error("Error fetching order book data: %s", e)
        return None

async def _generate_trading_decision(
self,:
symbol: str,
price: float,
volume: float,
ghost_state: GhostState,
order_book: OrderBookSnapshot,
walls: List[BuySellWall],
        profit_vector: Dict[str, Any],
brain_signal: Any,
mathematical_state: Dict[str, Any],
market_conditions: Dict[str, Any],
) -> Optional[TradingDecision]:"
        """Generate trading decision based on all available data."""
try:
            # Get brain trading decision
brain_decision = self.brain_engine.get_trading_decision(brain_signal)

# Calculate risk-adjusted position size
risk_metrics = self.risk_manager.calculate_position_size(
capital=self.state.current_capital,
price=price,'
volatility=market_conditions.get('volatility', 0.02),'
                confidence=brain_decision.get('confidence', 0.5),
)

# Determine action based on multiple factors
action = self._determine_action(
brain_decision=brain_decision,
ghost_state=ghost_state,
profit_vector=profit_vector,
walls=walls,
market_conditions=market_conditions,
)
'
if action == 'HOLD':
                return None

# Calculate optimal quantity
quantity = self._calculate_optimal_quantity(
action=action,
risk_metrics=risk_metrics,
brain_decision=brain_decision,
profit_vector=profit_vector,
order_book=order_book,
)

if quantity <= 0:
                return None

# Calculate execution price
execution_price = self._calculate_execution_price(
action=action, order_book=order_book, quantity=quantity
)

# Calculate confidence and risk scores
confidence = self._calculate_decision_confidence(
brain_decision=brain_decision,
ghost_state=ghost_state,
profit_vector=profit_vector,
mathematical_state=mathematical_state,
)

risk_score = self._calculate_risk_score(
action=action,
quantity=quantity,
market_conditions=market_conditions,
mathematical_state=mathematical_state,
)

# Create trading decision
decision = TradingDecision(
timestamp=time.time(),
symbol=symbol,
action=action,
quantity=quantity,
price=execution_price,
confidence=confidence,
strategy_branch=ghost_state.current_branch.value,'
                profit_potential=profit_vector.get('wall_enhanced_profit', 0.0),
                risk_score=risk_score,
exchange=list(self.ccxt_integration.exchanges.keys())[
0
],  # Use first exchange
granularity=order_book.granularity,
mathematical_state=mathematical_state,
market_conditions=market_conditions,
)

        return decision

        except Exception as e:"
            logger.error("Error generating trading decision: %s", e)
        return None

def _determine_action(
self,:
brain_decision: Dict[str, Any],
ghost_state: GhostState,
profit_vector: Dict[str, Any],
        walls: List[BuySellWall],
market_conditions: Dict[str, Any],
) -> str:"
        """Determine trading action based on multiple factors."""'
brain_action = brain_decision.get('action', 'HOLD')'
        brain_confidence = brain_decision.get('confidence', 0.0)

# Ghost state influence
ghost_confidence = ghost_state.confidence
ghost_profit_potential = ghost_state.profit_potential

# Profit vector influence'
        profit_potential = profit_vector.get('wall_enhanced_profit', 0.0)'
        pressure_ratio = profit_vector.get('pressure_ratio', 1.0)

# Market conditions influence'
volatility = market_conditions.get('volatility', 0.02)'
        momentum = market_conditions.get('momentum', 0.0)

# Combined decision logic'
if brain_action == 'HOLD' or brain_confidence < 0.3:'
            return 'HOLD'

# High confidence scenarios
if brain_confidence > 0.7 and ghost_confidence > 0.6:
            if profit_potential > 0.001:  # 0.1% profit potential
        return brain_action

# Moderate confidence with strong profit potential
        if brain_confidence > 0.5 and profit_potential > 0.002:  # 0.2% profit potential
        return brain_action

# Wall pressure influence'
if pressure_ratio > 1.5 and brain_action == 'BUY':'
            return 'BUY''
        elif pressure_ratio < 0.7 and brain_action == 'SELL':'
            return 'SELL'
'
        return 'HOLD'

def _calculate_optimal_quantity(
self,:
action: str,
risk_metrics: Dict[str, Any],
brain_decision: Dict[str, Any],
profit_vector: Dict[str, Any],
order_book: OrderBookSnapshot,
) -> float:"
        """Calculate optimal trading quantity."""
try:
            # Base quantity from risk metrics'
base_quantity = risk_metrics.get('position_size', 0.0)

# Adjust based on brain decision'
brain_position_size = brain_decision.get('position_size', 0.0)'
            brain_confidence = brain_decision.get('confidence', 0.5)

# Adjust based on profit potential'
            profit_potential = profit_vector.get('wall_enhanced_profit', 0.0)
            profit_multiplier = 1.0 + (profit_potential * 100)  # Scale profit potential

# Calculate final quantity
quantity = (
base_quantity
* brain_position_size
* profit_multiplier
* brain_confidence
)

# Apply limits
max_quantity = (
self.state.current_capital * 0.1 / order_book.mid_price
)  # Max 10% of capital
quantity = min(quantity, max_quantity)

# Minimum quantity check
min_quantity = 0.001  # Minimum BTC quantity
if quantity < min_quantity:
                return 0.0

        return quantity

        except Exception as e:"
            logger.error("Error calculating optimal quantity: %s", e)
        return 0.0

def _calculate_execution_price(:
self, action: str, order_book: OrderBookSnapshot, quantity: float
) -> float:"
        """Calculate execution price based on order book."""
try:'
            if action == 'BUY':
                # Use ask prices
orders = order_book.asks
else:  # SELL
# Use bid prices
orders = order_book.bids

if not orders:
                return order_book.mid_price

# Calculate weighted average price
total_volume = 0.0
            weighted_price = 0.0

for price, volume in orders:
                if total_volume >= quantity:
                    break

use_volume = min(volume, quantity - total_volume)
total_volume += use_volume
weighted_price += price * use_volume

if total_volume > 0:
                return weighted_price / total_volume
else:
                return order_book.mid_price

        except Exception as e:"
            logger.error("Error calculating execution price: %s", e)
        return order_book.mid_price

def _calculate_decision_confidence(
self,:
brain_decision: Dict[str, Any],
ghost_state: GhostState,
profit_vector: Dict[str, Any],
mathematical_state: Dict[str, Any],
) -> float:"
        """Calculate overall decision confidence."""
try:
            # Brain confidence'
brain_confidence = brain_decision.get('confidence', 0.5)

# Ghost state confidence
ghost_confidence = ghost_state.confidence

# Mathematical state confidence'
math_stability = mathematical_state.get('stability', 0.5)

# Profit vector confidence
            profit_confidence = min('
                1.0, profit_vector.get('wall_enhanced_profit', 0.0) * 1000
)

# Weighted average
confidence = (
brain_confidence * 0.4
                + ghost_confidence * 0.3
                + math_stability * 0.2
                + profit_confidence * 0.1
)

        return max(0.0, min(1.0, confidence))

        except Exception as e:"
            logger.error("Error calculating decision confidence: %s", e)
        return 0.5

def _calculate_risk_score(
self,:
action: str,
quantity: float,
market_conditions: Dict[str, Any],
mathematical_state: Dict[str, Any],
) -> float:"
        """Calculate risk score for the trading decision."""
try:
            risk_score = 0.0

# Volatility risk'
volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                risk_score += 0.3
            elif volatility > 0.03:  # Medium volatility
                risk_score += 0.2
else:  # Low volatility
risk_score += 0.1

# Position size risk'
position_value = quantity * market_conditions.get('price', 50000)
capital_ratio = position_value / self.state.current_capital
if capital_ratio > 0.1:  # More than 10% of capital
                risk_score += 0.3
            elif capital_ratio > 0.05:  # More than 5% of capital
                risk_score += 0.2
else:
                risk_score += 0.1

# Mathematical complexity risk'
complexity = mathematical_state.get('complexity', 0.5)
            if complexity > 0.8:  # High complexity
                risk_score += 0.2
            elif complexity > 0.6:  # Medium complexity
                risk_score += 0.1
else:
                risk_score += 0.05

        return min(1.0, risk_score)

        except Exception as e:"
            logger.error("Error calculating risk score: %s", e)
        return 0.5

def _update_pipeline_state(
self,:
decision: Optional[TradingDecision],
ghost_state: GhostState,
market_conditions: Dict[str, Any],
) -> None:"
        """Update pipeline state."""
self.state.timestamp = time.time()
self.state.active_strategy = ghost_state.current_branch
self.state.ghost_state = ghost_state'
self.state.market_volatility = market_conditions.get('volatility', 0.02)

if decision:
            self.trading_history.append(decision)

# Keep only recent history
if len(self.trading_history) > 1000:
                self.trading_history = self.trading_history[-500:]

async def execute_trade(self, decision: TradingDecision): -> Dict[str, Any]:"
        """Execute a trading decision."""
try:
            # Simulate trade execution (replace with actual exchange API calls)
execution_time = time.time()

# Calculate trade result
trade_value = decision.quantity * decision.price
commission = trade_value * 0.001  # 0.1% commission

# Update capital'
if decision.action == 'BUY':
                self.state.current_capital -= trade_value + commission
else:  # SELL
self.state.current_capital += trade_value - commission

# Update trade statistics
self.state.total_trades += 1

# Calculate profit (simplified)
            profit = 0.0'
            if decision.action == 'SELL':
                # Assume we bought at a lower price
profit = trade_value * 0.01  # 1% profit assumption

if profit > 0:
                self.state.winning_trades += 1

self.state.total_profit += profit

# Update Ghost Core performance
trade_result = {'
'profit': profit,'
'action': decision.action,'
'quantity': decision.quantity,'
'price': decision.price,'
'mathematical_state': decision.mathematical_state,
}

self.ghost_core.update_strategy_performance(
                decision.strategy_branch, trade_result
)

result = {'
'success': True,'
'execution_time': execution_time,'
'trade_value': trade_value,'
'commission': commission,'
'profit': profit,'
'new_capital': self.state.current_capital,'
'decision': decision,
}

            logger.info("
"✅ Trade executed: %s %.4f %s @ $%.2f (profit: $%.2f)",
decision.action,
decision.quantity,
decision.symbol,
decision.price,
profit,
)

        return result

        except Exception as e:"
            logger.error("Error executing trade: %s", e)'
        return {'success': False, 'error': str(e), 'decision': decision}

def get_pipeline_status(self) -> Dict[str, Any]:"
        """Get comprehensive pipeline status."""
        return {'
'state': {'
'current_capital': self.state.current_capital,'
'total_trades': self.state.total_trades,'
'winning_trades': self.state.winning_trades,'
'total_profit': self.state.total_profit,'
'win_rate': self.state.winning_trades / max(self.state.total_trades, 1),'
'active_strategy': self.state.active_strategy.value,'
'market_volatility': self.state.market_volatility,
},'
'ghost_core_status': self.ghost_core.get_system_status(),'
'brain_engine_metrics': self.brain_engine.get_metrics_summary(),'
'risk_manager_status': {'
                'current_risk_level': self.state.current_risk_level,'
                'risk_metrics': self.risk_manager.get_risk_metrics({}),
},'
'trading_history_size': len(self.trading_history),'
'market_data_history_size': len(self.market_data_history),
}

async def run_backtest(
self,:
price_series: List[Tuple[float, float]],  # (timestamp, price)"
symbol: str = "BTC/USDT",
) -> Dict[str, Any]:"
        """Run backtest on historical price data.""""
            logger.info("Starting backtest with %d price points", len(price_series))

initial_capital = self.state.current_capital
trades_executed = []

for i, (timestamp, price) in enumerate(price_series):
            # Simulate volume
volume = 1000 + (i % 100) * 10

# Determine granularity
granularity = 2 if price >= 10000 else 6

# Process market data
decision = await self.process_market_data(
symbol=symbol,
price=price,
volume=volume,
granularity=granularity,
tick_index=i,
)

# Execute trade if decision made
if decision:
                trade_result = await self.execute_trade(decision)
                trades_executed.append(trade_result)

# Log progress
if i % 100 == 0:
                logger.info("
"Backtest progress: %d/%d (%.1f%%)",
i,
len(price_series),
                    i / len(price_series) * 100,
)

# Calculate backtest results
final_capital = self.state.current_capital
total_return = (final_capital - initial_capital) / initial_capital

results = {'
'initial_capital': initial_capital,'
'final_capital': final_capital,'
'total_return': total_return,'
'total_trades': len(trades_executed),'
'winning_trades': sum('
1 for trade in trades_executed if trade.get('profit', 0) > 0
),'
'total_profit': sum(trade.get('profit', 0) for trade in trades_executed),'
'pipeline_status': self.get_pipeline_status(),
}

            logger.info("
"Backtest completed: %.2f%% return, %d trades",
total_return * 100,
len(trades_executed),
)
        return results

async def close(self) -> None:"
        """Close all connections and cleanup."""
try:
            await self.ccxt_integration.close_connections()"
            logger.info("Pipeline connections closed")
        except Exception as e:"
            logger.error("Error closing pipeline: %s", e)


async def demo_unified_pipeline():"
    """Demonstrate unified trading pipeline.""""
print("🚀 Unified Trading Pipeline Demo")"
print("=" * 50)

# Initialize pipeline
config = {'
'initial_capital': 100_000.0,'
'ghost_core': {'memory_depth': 100},'
'ccxt_integration': {'exchanges': ['binance']},'
'brain_trading_engine': {'confidence_threshold': 0.6},'
        'risk_manager': {'max_position_size': 0.1},
}

pipeline = UnifiedTradingPipeline(config)

try:
        # Generate test price series"
print("\nGenerating test price series...")
base_price = 50000.0
        price_series = []

for i in range(100):
            # Simulate price movement
price_change = np.random.normal(0, 0.01)  # 1% volatility
            base_price *= 1 + price_change
            price_series.append((time.time() + i, base_price))

# Run backtest"
print("Running backtest...")"
results = await pipeline.run_backtest(price_series, "BTC/USDT")

# Show results"
print("\nBacktest Results:")'"
print(f"  Initial Capital: ${results['initial_capital']:,.2f}")'"
print(f"  Final Capital: ${results['final_capital']:,.2f}")'"
print(f"  Total Return: {results['total_return']:.2%}")'"
print(f"  Total Trades: {results['total_trades']}")'"
print(f"  Winning Trades: {results['winning_trades']}")
print('"
f"  Win Rate: {results['winning_trades']/"'"
max(results['total_trades'], 1):.1%}""
)'"
print(f"  Total Profit: ${results['total_profit']:,.2f}")

# Show pipeline status'
status = results['pipeline_status']"
print("\nPipeline Status:")'"
print(f"  Active Strategy: {status['state']['active_strategy']}")'"
print(f"  Market Volatility: {status['state']['market_volatility']:.4f}")'"
print(f"  Ghost Core Memory: {status['ghost_core_status']['memory_depth']}")

        except Exception as e:"
        print(f"Demo failed: {e}")

finally:
        await pipeline.close()
"
print("\n✅ Unified Trading Pipeline demo completed!")

"
if __name__ == "__main__":
    asyncio.run(demo_unified_pipeline())
"
""""
"""'"