# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Demo Trading System - Schwabot UROS v1.0
=======================================

Simulates live trading using all mathematical functions and integrations.
Provides a complete demo environment for testing trading strategies without real money.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Import core components
try:
    from core.dlt_waveform_engine import DLTWaveformEngine, BitPhase as DLTBitPhase
    from core.matrix_mapper import MatrixMapper, BitPhase as MatrixBitPhase
    from core.profit_cycle_allocator import ProfitCycleAllocator
    from core.zpe_core import ZPECore
    from core.mathematical_integration_validator import MathematicalIntegrationValidator
    from core.ferris_rde_core import get_ferris_rde_core
    from core.tick_hash_processor import TickHashProcessor
    from core.unified_mathematics_config import get_unified_math
    from core.integrated_alif_aleph_system import IntegratedAlifAlephSystem
    from core.real_trading_integration import get_real_trading_integration
CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
logger.error(f"Critical core component missing: {e}")
    raise RuntimeError(f"Required core component not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DemoMarketData:
    """Simulated market data for demo trading."""
symbol: str
price: float
volume: float
timestamp: datetime
volatility: float
entropy_level: float
complexity: float
trend_strength: float
market_heat: float
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DemoTrade:
    """Demo trade execution."""
trade_id: str
symbol: str
side: str  # "buy" or "sell"
quantity: float
price: float
timestamp: datetime
tensor_score: float
bit_phase: int
basket_id: Optional[str] = None
profit: float = 0.0
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DemoPortfolio:
    """Demo portfolio state."""
total_value: float
cash: float
positions: Dict[str, float]
trades: List[DemoTrade]
total_profit: float
total_trades: int
win_rate: float
timestamp: datetime
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DemoStrategy:
    """Demo trading strategy configuration."""
strategy_id: str
name: str
symbols: List[str]
initial_capital: float
risk_tolerance: float
max_position_size: float
stop_loss_pct: float
take_profit_pct: float
enabled: bool = True
metadata: Dict[str, Any] = field(default_factory=dict)

class DemoMarketSimulator:
    """Simulates market data for demo trading."""

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC/USDC', 'ETH/USDC', 'ADA/USDC', 'DOT/USDC']
self.base_prices = {
'BTC/USDC': 50000.0,
'ETH/USDC': 3000.0,
'ADA/USDC': 0.5,
'DOT/USDC': 7.0
}
self.current_prices = self.base_prices.copy()
        self.volatility = {symbol: 0.02 for symbol in self.symbols}
self.trend_direction = {symbol: 1.0 for symbol in self.symbols}

        # Market state
self.market_heat = 0.5
self.entropy_level = 4.0
self.complexity = 0.6

logger.info(f"Demo market simulator initialized with {len(self.symbols)} symbols")

    def generate_market_data(self, symbol: str) -> DemoMarketData:
        """Generate simulated market data for a symbol."""
        try:
            # Update price with random walk
current_price = self.current_prices[symbol]
volatility = self.volatility[symbol]
trend = self.trend_direction[symbol]

            # Random price movement
price_change = np.random.normal(0, volatility) * current_price
            trend_change = trend * volatility * 0.1 * current_price
total_change = price_change + trend_change

            # Update price
new_price = current_price + total_change
self.current_prices[symbol] = unified_math.max(new_price, current_price * 0.5)  # Prevent negative prices

            # Update trend direction occasionally
            if np.random.random() < 0.01:  # 1% chance to change trend
                self.trend_direction[symbol] *= -1

            # Generate volume
base_volume = 1000.0
volume_variation = np.random.normal(1.0, 0.3)
            volume = base_volume * volume_variation * (1 + unified_math.abs(price_change) / current_price)

            # Update market state
self.market_heat = np.clip(self.market_heat + np.random.normal(0, 0.01), 0.0, 1.0)
            self.entropy_level = np.clip(self.entropy_level + np.random.normal(0, 0.1), 1.0, 8.0)
            self.complexity = np.clip(self.complexity + np.random.normal(0, 0.02), 0.1, 1.0)

            return DemoMarketData(
                symbol=symbol,
price=self.current_prices[symbol],
volume=volume,
timestamp=datetime.now(),
                volatility=volatility,
entropy_level=self.entropy_level,
complexity=self.complexity,
trend_strength=unified_math.abs(trend),
                market_heat=self.market_heat


        except Exception as e:
logger.error(f"Error generating market data for {symbol}: {e}")
            return None

    def get_all_market_data(self) -> Dict[str, DemoMarketData]:
        """Get market data for all symbols."""
market_data = {}
        for symbol in self.symbols:
data = self.generate_market_data(symbol)
            if data:
market_data[symbol] = data
        return market_data

class DemoTradingSystem:
    """
Demo trading system that simulates live trading using all mathematical functions.

Features:
- Real-time market data simulation
- Mathematical function integration
- Portfolio tracking and management
- Performance analytics
- Risk management
- Strategy backtesting
"""

    def __init__(self, config_path: str = "./config/demo_trading_system_config.json"):
        self.config_path = config_path
self.config = self._load_configuration()

        # Initialize real core components
self._initialize_core_components()

        # Trading state
self.is_running: bool = False
self.current_portfolio: Dict[str, Any] = {}
self.trade_history: List[Dict[str, Any]] = []
self.performance_metrics: Dict[str, Any] = {}

        # Market simulation
self.market_simulator = DemoMarketSimulator(self.config.get("market_simulation", {}))

        # Threading
self.executor = ThreadPoolExecutor(max_workers=4)
        self.stop_event = threading.Event()

logger.info("Demo Trading System initialized with real core components")

    def _initialize_core_components(self) -> None:
        """Initialize all core components with real implementations."""
        try:
            # Initialize core components
self.dlt_engine = DLTWaveformEngine()
            self.matrix_mapper = MatrixMapper()
            self.profit_allocator = ProfitCycleAllocator()
            self.zpe_core = ZPECore()
            self.math_validator = MathematicalIntegrationValidator()
            self.ferris_rde = get_ferris_rde_core()
            self.tick_processor = TickHashProcessor()
            self.unified_math = get_unified_math()
            self.alif_aleph_system = IntegratedAlifAlephSystem()
            self.trading_integration = get_real_trading_integration()

logger.info("✅ All core components initialized successfully")

        except Exception as e:
logger.error(f"❌ Failed to initialize core components: {e}")
            raise RuntimeError(f"Core component initialization failed: {e}")

    def add_strategy(self, strategy: DemoStrategy) -> None:
        """Add a trading strategy to the demo system."""
self.strategies[strategy.strategy_id] = strategy
logger.info(f"Added strategy: {strategy.name}")

    def start_trading(self) -> None:
        """Start the demo trading system."""
        if self.is_running:
logger.warning("Demo trading system is already running")
            return

self.is_running = True
self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()

logger.info("Demo trading system started")

    def stop_trading(self) -> None:
        """Stop the demo trading system."""
self.is_running = False
        if self.trading_thread:
self.trading_thread.join(timeout=5.0)

logger.info("Demo trading system stopped")

    def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.is_running:
            try:
start_time = time.time()

                # Generate market data
market_data = self.market_simulator.get_all_market_data()

                # Process each symbol
                for symbol, data in market_data.items():
                    self._process_symbol(symbol, data)

                # Sleep for tick interval
elapsed = time.time() - start_time
                sleep_time = unified_math.max(0, self.tick_interval - elapsed)
                if sleep_time > 0:
time.sleep(sleep_time)

            except Exception as e:
logger.error(f"Error in trading loop: {e}")
                time.sleep(1.0)

    def _process_symbol(self, symbol: str, market_data: DemoMarketData) -> None:
        """Process a single symbol for trading decisions."""
        try:
            # Generate waveform data from price movement
price_history = self._get_price_history(symbol)
            if len(price_history) < 100:
                return

            # Process waveform
            if self.dlt_engine:
waveform_result = self.dlt_engine.process_waveform_data(
                    name=f"{symbol}_waveform",
x=np.array(price_history),
                    sample_rate=1.0


                if waveform_result.get('success'):
                    # Get tensor score
tensor_score = waveform_result.get('tensor_score', 0.0)

                    # Make trading decision
self._make_trading_decision(symbol, market_data, tensor_score)

        except Exception as e:
logger.error(f"Error processing symbol {symbol}: {e}")

    def _get_price_history(self, symbol: str) -> List[float]:
        """Get price history for a symbol."""
        # In a real implementation, this would fetch from a database
        # For demo, we'll generate synthetic price history
base_price = self.market_simulator.base_prices.get(symbol, 100.0)
        history = []

        for i in range(100):
            # Generate price with some trend and noise
trend = np.unified_math.sin(i * 0.1) * 0.01
            noise = np.random.normal(0, 0.005)
            price = base_price * (1 + trend + noise)
            history.append(price)

        return history

    def _make_trading_decision(self, symbol: str, market_data: DemoMarketData, tensor_score: float) -> None:
        """Make trading decision based on mathematical analysis."""
        try:
            # Determine bit phase
bit_phase = self._determine_bit_phase(market_data)

            # Calculate position size based on tensor score and risk
position_size = self._calculate_position_size(tensor_score, bit_phase)

            if position_size > 0:
                # Determine trade direction
                if tensor_score > 0.3:
side = "buy"
                elif tensor_score < -0.3:
side = "sell"
                else:
                    return  # No trade

                # Execute trade
self._execute_trade(symbol, side, position_size, market_data.price, tensor_score, bit_phase)

        except Exception as e:
logger.error(f"Error making trading decision for {symbol}: {e}")

    def _determine_bit_phase(self, market_data: DemoMarketData) -> int:
        """Determine optimal bit phase based on market conditions."""
        try:
entropy_level = market_data.entropy_level
complexity = market_data.complexity
volatility = market_data.volatility

            # Calculate composite score
composite_score = (entropy_level * 0.4 + complexity * 0.3 + volatility * 100 * 0.3)

            # Determine bit phase based on composite score
            if composite_score < 2.0:
                return 4  # 4-bit conservative
            elif composite_score < 5.0:
                return 8  # 8-bit balanced
            else:
                return 42  # 42-bit quantum

        except Exception as e:
logger.error(f"Error determining bit phase: {e}")
            return 8  # Default to 8-bit

    def _calculate_position_size(self, tensor_score: float, bit_phase: int) -> float:
        """Calculate position size based on tensor score and bit phase."""
        try:
            # Base position size
base_size = self.current_capital * 0.01  # 1% of capital

            # Adjust based on tensor score
tensor_factor = unified_math.abs(tensor_score)

            # Adjust based on bit phase
            if bit_phase == 4:
bit_factor = 0.5  # Conservative
            elif bit_phase == 8:
bit_factor = 1.0  # Balanced
            else:  # 42-bit
bit_factor = 1.5  # Aggressive

            # Calculate final position size
position_size = base_size * tensor_factor * bit_factor

            # Apply risk management
max_position = self.current_capital * 0.1  # Max 10% of capital
position_size = unified_math.min(position_size, max_position)

            return position_size

        except Exception as e:
logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _execute_trade(self, symbol: str, side: str, quantity: float, price: float,
                      tensor_score: float, bit_phase: int) -> None:
"""Execute a demo trade."""
        try:
            # Generate real tick hash for the trade
tick_hash = self.tick_processor.generate_tick_hash(
                price=price,
volume=quantity * price,
timestamp=time.time()


            # Process through Ferris RDE for 16-bit mapping
price_mapping = self.ferris_rde.map_btc_price_16bit(price)

            # Calculate tensor score using real matrix mapping
tensor_score = self.matrix_mapper.calculate_tensor_score(
                price=price,
volume=quantity * price,
market_data={
"mapped_16bit": price_mapping.mapped_price,
"ferris_phase": self.ferris_rde.current_phase.value,
"volatility": np.random.uniform(0.01, 0.05),
                    "entropy_level": np.random.uniform(1.0, 8.0)
                }


            # Determine bit phase using real bit phase engine
bit_phase = self.matrix_mapper.resolve_bit_phase(
                tick_hash,
price_mapping.mapped_price


            # Use DLT engine for trade analysis
dlt_analysis = self.dlt_engine.analyze_tick_for_decision(
                price=price,
volume=quantity * price,
tensor_score=tensor_score,
bit_phase=bit_phase


            # Calculate trade confidence using unified mathematics
confidence = self.unified_math.execute_with_monitoring(
                "trade_confidence",
self._calculate_trade_confidence,
tensor_score, bit_phase, dlt_analysis


            # Execute trade through real trading integration
trade_result = self.trading_integration.execute_trade(
                symbol=symbol,
side=side,
quantity=quantity,
price=price,
tensor_score=tensor_score,
bit_phase=bit_phase,
confidence=confidence


            # Update portfolio using real profit allocation
self._update_portfolio(trade_result, tensor_score, bit_phase)

            # Record trade with real metadata
trade_record = {
"trade_id": trade_result.get("trade_id", f"demo_trade_{len(self.trade_history)}"),
                "timestamp": datetime.now(),
                "symbol": symbol,
"side": side,
"quantity": quantity,
"price": price,
"tensor_score": tensor_score,
"bit_phase": bit_phase,
"confidence": confidence,
"dlt_analysis": dlt_analysis,
"tick_hash": tick_hash,
"mapped_16bit": price_mapping.mapped_price,
"ferris_phase": self.ferris_rde.current_phase.value,
"status": trade_result.get("status", "executed")
            }

self.trade_history.append(trade_record)

logger.info(f"✅ Trade executed: {symbol} {side} {quantity} @ {price}")

        except Exception as e:
logger.error(f"❌ Error executing trade: {e}")
            raise RuntimeError(f"Trade execution failed: {e}")

    def _calculate_trade_confidence(self, tensor_score: float, bit_phase: int, dlt_analysis: Dict[str, Any]) -> float:
        """Calculate trade confidence using mathematical models."""
        try:
            # Base confidence from tensor score
base_confidence = tensor_score

            # Bit phase adjustment
bit_phase_adjustment = unified_math.min(bit_phase / 16.0, 1.0)

            # DLT analysis adjustment
dlt_score = dlt_analysis.get("waveform_score", 0.5)

            # Combine using weighted average
confidence = (
                base_confidence * 0.4 +
bit_phase_adjustment * 0.3 +
dlt_score * 0.3


            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
logger.error(f"Error calculating trade confidence: {e}")
            return 0.5

    def _update_portfolio(self, trade_result: Dict[str, Any], tensor_score: float, bit_phase: int) -> None:
        """Update portfolio using real profit allocation logic."""
        try:
            # Use profit cycle allocator for portfolio updates
portfolio_update = self.profit_allocator.calculate_portfolio_update(
                trade_result=trade_result,
tensor_score=tensor_score,
bit_phase=bit_phase,
current_portfolio=self.current_portfolio


            # Apply portfolio update
self.current_portfolio.update(portfolio_update)

            # Update performance metrics
self._update_performance_metrics(trade_result, tensor_score, bit_phase)

        except Exception as e:
logger.error(f"Error updating portfolio: {e}")

    def _update_performance_metrics(self, trade_result: Dict[str, Any], tensor_score: float, bit_phase: int) -> None:
        """Update performance metrics using mathematical models."""
        try:
            # Calculate trade performance
trade_pnl = trade_result.get("realized_pnl", 0.0)

            # Update metrics using unified mathematics
self.performance_metrics = self.unified_math.execute_with_monitoring(
                "performance_update",
self._calculate_performance_metrics,
trade_pnl, tensor_score, bit_phase, self.performance_metrics


        except Exception as e:
logger.error(f"Error updating performance metrics: {e}")

    def _calculate_performance_metrics(self, trade_pnl: float, tensor_score: float, bit_phase: int, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics using mathematical models."""
        try:
            # Initialize metrics if not present
            if not current_metrics:
current_metrics = {
"total_trades": 0,
"winning_trades": 0,
"total_pnl": 0.0,
"win_rate": 0.0,
"average_confidence": 0.0,
"average_tensor_score": 0.0
}

            # Update metrics
current_metrics["total_trades"] += 1
current_metrics["total_pnl"] += trade_pnl

            if trade_pnl > 0:
current_metrics["winning_trades"] += 1

            # Calculate averages
total_trades = current_metrics["total_trades"]
current_metrics["win_rate"] = current_metrics["winning_trades"] / total_trades

            # Update running averages for confidence and tensor score
current_avg_confidence = current_metrics.get("average_confidence", 0.0)
            current_avg_tensor = current_metrics.get("average_tensor_score", 0.0)

            # Calculate new averages (simplified - in real implementation would use proper running average)
            confidence = unified_math.max(0.0, unified_math.min(1.0, tensor_score))  # Use tensor score as proxy for confidence
            current_metrics["average_confidence"] = (current_avg_confidence * (total_trades - 1) + confidence) / total_trades
            current_metrics["average_tensor_score"] = (current_avg_tensor * (total_trades - 1) + tensor_score) / total_trades

            return current_metrics

        except Exception as e:
logger.error(f"Error calculating performance metrics: {e}")
            return current_metrics

    def get_portfolio_status(self) -> DemoPortfolio:
        """Get current portfolio status."""
        try:
            # Calculate current portfolio value
portfolio_value = self.current_capital

            # Add value of positions
            for symbol, quantity in self.positions.items():
                if quantity > 0:
current_price = self.market_simulator.current_prices.get(symbol, 0.0)
                    portfolio_value += quantity * current_price

            # Calculate profit
total_profit = portfolio_value - self.initial_capital

            # Calculate win rate
            if self.trades:
profitable_trades = sum(1 for trade in self.trades if trade.profit > 0)
                win_rate = profitable_trades / len(self.trades)
            else:
win_rate = 0.0

            return DemoPortfolio(
                total_value=portfolio_value,
cash=self.current_capital,
positions=self.positions.copy(),
                trades=self.trades.copy(),
                total_profit=total_profit,
total_trades=len(self.trades),
                win_rate=win_rate,
timestamp=datetime.now()


        except Exception as e:
logger.error(f"Error getting portfolio status: {e}")
            return None

    def run_mathematical_validation(self) -> Dict[str, Any]:
        """Run mathematical validation on the demo system."""
        try:
            if not CORE_COMPONENTS_AVAILABLE:
                return {'error': 'Core components not available'}

validator = MathematicalIntegrationValidator()
            results = validator.run_comprehensive_validation()

            # Store results
self.mathematical_validation_results.append({
                'timestamp': datetime.now().isoformat(),
                'results': results
})

            return results

        except Exception as e:
logger.error(f"Error running mathematical validation: {e}")
            return {'error': str(e)}

    def export_demo_results(self, output_path: str = "demo_trading_results.json") -> None:
        """Export demo trading results."""
        try:
portfolio = self.get_portfolio_status()

results_data = {
'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
'portfolio': {
'total_value': portfolio.total_value,
'cash': portfolio.cash,
'total_profit': portfolio.total_profit,
'total_trades': portfolio.total_trades,
'win_rate': portfolio.win_rate,
'positions': portfolio.positions
},
'trades': [
{
'trade_id': trade.trade_id,
'symbol': trade.symbol,
'side': trade.side,
'quantity': trade.quantity,
'price': trade.price,
'timestamp': trade.timestamp.isoformat(),
                        'tensor_score': trade.tensor_score,
'bit_phase': trade.bit_phase,
'profit': trade.profit
}
                    for trade in self.trades
],
'mathematical_validation': self.mathematical_validation_results
}

            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

safe_print(f"✅ Demo results exported to {output_path}")

        except Exception as e:
safe_print(f"❌ Error exporting demo results: {e}")

def create_demo_strategy(strategy_id: str, name: str, symbols: List[str],
                        initial_capital: float) -> DemoStrategy:
"""Create a demo trading strategy."""
    return DemoStrategy(
        strategy_id=strategy_id,
name=name,
symbols=symbols,
initial_capital=initial_capital,
risk_tolerance=0.1,
max_position_size=0.1,
stop_loss_pct=0.05,
take_profit_pct=0.1


def main():
    """Main function to run demo trading system."""
safe_print("🚀 Starting Demo Trading System...")

    # Create demo trading system
demo_system = DemoTradingSystem(initial_capital=100000.0)

    # Add strategies
strategy1 = create_demo_strategy(
        strategy_id="strategy_1",
name="Conservative BTC Strategy",
symbols=['BTC/USDC'],
initial_capital=50000.0

demo_system.add_strategy(strategy1)

strategy2 = create_demo_strategy(
        strategy_id="strategy_2",
name="Multi-Asset Strategy",
symbols=['BTC/USDC', 'ETH/USDC', 'ADA/USDC'],
initial_capital=50000.0

demo_system.add_strategy(strategy2)

    # Start trading
demo_system.start_trading()

    try:
        # Run for 60 seconds
safe_print("📈 Demo trading running for 60 seconds...")
        time.sleep(60)

        # Stop trading
demo_system.stop_trading()

        # Get results
portfolio = demo_system.get_portfolio_status()
        safe_print("\n📊 DEMO TRADING RESULTS")
        safe_print(f"Initial Capital: ${demo_system.initial_capital:,.2f}")
        safe_print(f"Final Portfolio Value: ${portfolio.total_value:,.2f}")
        safe_print(f"Total Profit: ${portfolio.total_profit:,.2f}")
        safe_print(f"Total Trades: {portfolio.total_trades}")
        safe_print(f"Win Rate: {portfolio.win_rate:.2%}")

        # Run mathematical validation
safe_print("\n🧪 Running Mathematical Validation...")
        validation_results = demo_system.run_mathematical_validation()
        safe_print(f"Validation Status: {validation_results.get('overall_status', 'UNKNOWN')}")

        # Export results
demo_system.export_demo_results()

    except KeyboardInterrupt:
safe_print("\n⏹️ Demo trading stopped by user")
        demo_system.stop_trading()

    return 0

if __name__ == "__main__":
exit(main())
