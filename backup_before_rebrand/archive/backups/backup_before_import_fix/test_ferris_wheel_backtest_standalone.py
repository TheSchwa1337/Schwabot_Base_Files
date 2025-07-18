import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from core.math.ferris_wheel_rde import FerrisWheelRDE

"""
Ferris Wheel RDE Standalone Backtesting System
==============================================

Comprehensive backtesting framework for the Ferris Wheel RDE system.
Tests mathematical logic, strategy performance, risk metrics, and provides
validation for live trading readiness.

Features:
- Historical price data simulation
- Strategy performance tracking
- Risk metrics calculation (Sharpe ratio, max drawdown, etc.)
- Mathematical validation
- Live trading readiness assessment
- Performance visualization
"""



# Import Ferris Wheel RDE directly

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid core module issues

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
    class BacktestResult:
    """Results from a backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    strategy_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    trade_history: List[Dict[str, Any]]
    mathematical_validation: Dict[str, bool]
    live_ready_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
    class TradeRecord:
    """Record of a single trade."""

    timestamp: float
    price: float
    strategy: str
    bit_mode: int
    phase: str
    probability: float
    entropy: float
    action: str  # 'buy', 'sell', 'hold'
    pnl: float = 0.0
    cumulative_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FerrisWheelBacktester:
    """
    Comprehensive backtesting system for Ferris Wheel RDE.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.ferris_rde = FerrisWheelRDE()
        self.trade_history: List[TradeRecord] = []
        self.performance_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.mathematical_checks: Dict[str, bool] = {}

        # Risk management parameters
        self.max_position_size = 0.1  # 10% of balance
        self.stop_loss = 0.5  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit

        logger.info()
            f"🎯 Ferris Wheel Backtester initialized with ${initial_balance:,.2f}"
        )

    def generate_historical_data():-> List[Tuple[float, float]]:
        """
        Generate realistic historical price data for backtesting.

        Args:
            days: Number of days to simulate
            volatility: Daily volatility (default 2%)

        Returns:
            List of (timestamp, price) tuples
        """
        logger.info(f"📊 Generating {days} days of historical data...")

        # Start with realistic BTC price
        base_price = 45000.0
        prices = []
        current_price = base_price

        # Generate timestamps (hourly, data)
        start_time = time.time() - (days * 24 * 3600)

        for hour in range(days * 24):
            timestamp = start_time + (hour * 3600)

            # Add market cycles (trend + noise + volatility)
            trend = 0.001 * math.sin(hour / (24 * 7))  # Weekly cycle
            noise = random.gauss(0, volatility / math.sqrt(24))  # Hourly noise
            volatility_shock = ()
                random.gauss(0, volatility) * random.random()
            )  # Occasional shocks

            # Price change
            change = trend + noise + volatility_shock
            current_price *= 1 + change

            # Ensure price stays reasonable
            current_price = max(1000, min(100000, current_price))

            prices.append((timestamp, current_price))

        logger.info()
            f"📈 Generated {len(prices)} price points, final price: ${current_price:,.2f}"
        )
        return prices

    def execute_trade():-> TradeRecord:
        """
        Execute a trade based on RDE decision.

        Args:
            price: Current price
            strategy: Selected strategy
            probability: Strategy probability
            bit_mode: Bit mode used
            phase: Market phase
            entropy: Entropy level

        Returns:
            TradeRecord with trade details
        """
        # Determine action based on strategy and probability
        if strategy == "hold":
            action = "hold"
        elif strategy == "flip" and probability > 0.6:
            action = "sell" if random.random() > 0.5 else "buy"
        elif strategy == "exit" and probability > 0.7:
            action = "sell"
        elif strategy == "entry" and probability > 0.6:
            action = "buy"
        elif strategy == "stable_swap" and probability > 0.5:
            action = "buy" if random.random() > 0.5 else "sell"
        else:
            action = "hold"

        # Calculate position size based on probability and risk management
        if action != "hold":
            position_size = min(self.max_position_size, probability * 0.2)
            position_value = self.balance * position_size
        else:
            position_size = 0.0
            position_value = 0.0

        # Simulate P&L (simplified - in real trading this would be more, complex)
        pnl = 0.0
        if action == "buy" and position_value > 0:
            # Simulate price movement
            price_change = random.gauss(0.01, 0.05)  # Small random movement
            pnl = position_value * price_change
        elif action == "sell" and position_value > 0:
            price_change = random.gauss(-0.01, 0.05)
            pnl = position_value * price_change

        # Update balance
        self.balance += pnl

        # Create trade record
        trade = TradeRecord()
            timestamp=time.time(),
            price=price,
            strategy=strategy,
            bit_mode=bit_mode,
            phase=phase,
            probability=probability,
            entropy=entropy,
            action=action,
            pnl=pnl,
            cumulative_pnl=self.balance - self.initial_balance,
            metadata={"position_size": position_size, "position_value": position_value},
        )

        self.trade_history.append(trade)
        self.performance_history.append(self.balance)

        return trade

    def calculate_risk_metrics():-> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if not self.performance_history:
            return {}

        returns = np.diff(self.performance_history) / self.performance_history[:-1]

        # Basic metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        avg_return = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0

        # Maximum drawdown
        peak = self.performance_history[0]
        max_dd = 0.0
        for value in self.performance_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Win rate
        winning_trades = sum(1 for trade in self.trade_history if trade.pnl > 0)
        total_trades = len([t for t in self.trade_history if t.action != "hold"])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(trade.pnl for trade in self.trade_history if trade.pnl > 0)
        gross_loss = abs()
            sum(trade.pnl for trade in self.trade_history if trade.pnl < 0)
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")"

        return {}
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "volatility": volatility,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t.pnl for t in self.trade_history])
            if self.trade_history
            else 0.0,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
        }

    def validate_mathematics():-> Dict[str, bool]:
        """Validate mathematical components of the RDE system."""
        checks = {}

        # Check entropy calculation
        test_string = "test_entropy_string"
        entropy = self.ferris_rde._calculate_entropy(test_string)
        checks["entropy_calculation"] = ()
            0.0 <= entropy <= 4.0
        )  # Reasonable bounds for this string

        # Check weight normalization
        test_weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        normalized = self.ferris_rde.normalize_weights(test_weights)
        checks["weight_normalization"] = abs(sum(normalized.values()) - 1.0) < 1e-6

        # Check softmax
        softmax_result = self.ferris_rde.softmax(test_weights)
        checks["softmax_calculation"] = abs(sum(softmax_result.values()) - 1.0) < 1e-6

        # Check phase detection
        phase = self.ferris_rde.get_phase_state(50000.0)
        checks["phase_detection"] = phase in ["ascent", "peak", "descent", "trough"]

        # Check strategy mutation
        original_perf = self.ferris_rde.strategy_performance.copy()
        self.ferris_rde.mutate_strategy_weights({"hold": 1.5})
        checks["strategy_mutation"] = any()
            self.ferris_rde.strategy_performance[k] != original_perf[k]
            for k in original_perf
        )

        # Check reinforcement learning
        original_perf = self.ferris_rde.strategy_performance["hold"]
        self.ferris_rde.update_strategy_reward("hold", 0.5)
        checks["reinforcement_learning"] = ()
            self.ferris_rde.strategy_performance["hold"] != original_perf
        )

        self.mathematical_checks = checks
        return checks

    def calculate_live_ready_score():-> float:
        """Calculate a score indicating readiness for live trading."""
        if not self.trade_history:
            return 0.0

        risk_metrics = self.calculate_risk_metrics()
        math_validation = self.validate_mathematics()

        # Score components (0-1 each)
        score_components = []

        # Performance score (30% weight)
        if risk_metrics.get("total_return", 0) > 0:
            score_components.append(0.3)
        elif risk_metrics.get("total_return", 0) > -0.1:
            score_components.append(0.2)
        else:
            score_components.append(0.0)

        # Risk management score (25% weight)
        if risk_metrics.get("max_drawdown", 1) < 0.1:
            score_components.append(0.25)
        elif risk_metrics.get("max_drawdown", 1) < 0.2:
            score_components.append(0.15)
        else:
            score_components.append(0.0)

        # Mathematical validation score (25% weight)
        math_score = ()
            sum(math_validation.values()) / len(math_validation)
            if math_validation
            else 0.0
        )
        score_components.append(0.25 * math_score)

        # Consistency score (20% weight)
        if risk_metrics.get("win_rate", 0) > 0.5:
            score_components.append(0.2)
        elif risk_metrics.get("win_rate", 0) > 0.4:
            score_components.append(0.1)
        else:
            score_components.append(0.0)

        return sum(score_components)

    def run_backtest():-> BacktestResult:
        """
        Run comprehensive backtest.

        Args:
            days: Number of days to backtest
            volatility: Market volatility

        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"🚀 Starting Ferris Wheel RDE backtest for {days} days...")

        # Reset state
        self.balance = self.initial_balance
        self.trade_history = []
        self.performance_history = [self.initial_balance]
        self.ferris_rde = FerrisWheelRDE()  # Fresh instance

        # Generate historical data
        price_data = self.generate_historical_data(days, volatility)

        # Run backtest
        for i, (timestamp, price) in enumerate(price_data):
            # Run RDE cycle
            result = self.ferris_rde.ferris_rde_cycle()
                price,
                bit_mode=4 if i % 3 == 0 else 8 if i % 3 == 1 else 42,
                timestamp=timestamp,
            )

            # Execute trade
            trade = self.execute_trade()
                price=price,
                strategy=result["strategy"],
                probability=result["probability"],
                bit_mode=result["ferris_state"].bit_state,
                phase=result["phase"],
                entropy=result["ferris_state"].entropy_level,
            )

            # Periodic strategy updates
            if i % 24 == 0 and i > 0:  # Daily updates
                # Simulate strategy performance feedback
                success_scores = {trade.strategy: random.uniform(0.5, 1.5)}
                self.ferris_rde.mutate_strategy_weights(success_scores)

            if i % 12 == 0 and i > 0:  # Twice daily reinforcement
                reward = random.uniform(-0.3, 0.7)
                self.ferris_rde.update_strategy_reward(trade.strategy, reward)

            # Progress logging
            if i % (len(price_data) // 10) == 0:
                logger.info()
                    f"📊 Backtest progress: {i}/{len(price_data)} ({i / len(price_data) * 100:.1f}%)"
                )

        # Calculate results
        risk_metrics = self.calculate_risk_metrics()
        mathematical_validation = self.validate_mathematics()
        live_ready_score = self.calculate_live_ready_score()

        # Create trade history for result
        trade_history = []
            {}
                "timestamp": trade.timestamp,
                "price": trade.price,
                "strategy": trade.strategy,
                "action": trade.action,
                "pnl": trade.pnl,
                "probability": trade.probability,
                "entropy": trade.entropy,
            }
            for trade in self.trade_history
        ]
        result = BacktestResult()
            total_trades=risk_metrics.get("total_trades", 0),
            winning_trades=risk_metrics.get("winning_trades", 0),
            losing_trades=risk_metrics.get("total_trades", 0)
            - risk_metrics.get("winning_trades", 0),
            win_rate=risk_metrics.get("win_rate", 0.0),
            total_return=risk_metrics.get("total_return", 0.0),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", 0.0),
            max_drawdown=risk_metrics.get("max_drawdown", 0.0),
            strategy_performance=self.ferris_rde.strategy_performance,
            risk_metrics=risk_metrics,
            trade_history=trade_history,
            mathematical_validation=mathematical_validation,
            live_ready_score=live_ready_score,
        )

        logger.info(f"✅ Backtest completed! Final balance: ${self.balance:,.2f}")
        logger.info()
            f"📈 Total return: {risk_metrics.get('total_return', 0) * 100:.2f}%"
        )
        logger.info(f"🎯 Live ready score: {live_ready_score:.2f}/1.0")

        return result

    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to JSON file."""
        # Convert dataclass to dict for JSON serialization
        result_dict = {}
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "strategy_performance": result.strategy_performance,
            "risk_metrics": result.risk_metrics,
            "trade_history": result.trade_history,
            "mathematical_validation": result.mathematical_validation,
            "live_ready_score": result.live_ready_score,
            "timestamp": result.timestamp,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"💾 Backtest results saved to {filepath}")


def main():
    """Run comprehensive backtest demonstration."""
    print("🎯 Ferris Wheel RDE Backtesting System")
    print("=" * 60)

    # Initialize backtester
    backtester = FerrisWheelBacktester(initial_balance=10000.0)

    # Run backtest
    result = backtester.run_backtest(days=90, volatility=0.2)

    # Print results
    print("\n📊 Backtest Results Summary:")
    print("-" * 40)
    print(f"Initial Balance: ${backtester.initial_balance:,.2f}")
    print(f"Final Balance: ${backtester.balance:,.2f}")
    print(f"Total Return: {result.total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Win Rate: {result.win_rate * 100:.1f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Live Ready Score: {result.live_ready_score:.2f}/1.0")

    print("\n🔬 Mathematical Validation:")
    print("-" * 40)
    for check, passed in result.mathematical_validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")

    print("\n📈 Strategy Performance:")
    print("-" * 40)
    for strategy, performance in result.strategy_performance.items():
        print(f"{strategy}: {performance:.4f}")

    # Save results
    backtester.save_results(result, "ferris_wheel_backtest_results.json")

    # Live trading readiness assessment
    print("\n🎯 Live Trading Readiness Assessment:")
    print("-" * 40)
    if result.live_ready_score >= 0.8:
        print("🟢 EXCELLENT - Ready for live trading")
    elif result.live_ready_score >= 0.6:
        print("🟡 GOOD - Consider with caution")
    elif result.live_ready_score >= 0.4:
        print("🟠 FAIR - Needs improvement")
    else:
        print("🔴 POOR - Not ready for live trading")

    print("\nDetailed score breakdown:")
    print(f"  Performance: {result.total_return * 100:.2f}% return")
    print(f"  Risk Management: {result.max_drawdown * 100:.2f}% max drawdown")
    print()
        f"  Mathematical Validation: {sum(result.mathematical_validation.values())}/{len(result.mathematical_validation)} checks passed"
    )
    print(f"  Consistency: {result.win_rate * 100:.1f}% win rate")

    # Print sample trade history
    print("\n📋 Sample Trade History (Last 10):")
    print("-" * 60)
    for i, trade in enumerate(result.trade_history[-10:]):
        print()
            f"{i + 1:2d}. {trade['strategy']:<12} | {trade['action']:<4} | "
            f"Price: ${trade['price']:,.2f} | PnL: ${trade['pnl']:+.2f} | "
            f"Prob: {trade['probability']:.3f} | Entropy: {trade['entropy']:.3f}"
        )


if __name__ == "__main__":
    main()
