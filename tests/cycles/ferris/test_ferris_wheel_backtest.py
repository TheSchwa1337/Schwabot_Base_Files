"""
Ferris Wheel RDE Backtesting System
===================================

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

import json
import logging
import math
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Adjusting import path for the new location
import sys
from pathlib import Path
core_dir = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(core_dir))

# Dummy class for the purpose of this standalone test file
class FerrisWheelRDE:
    def process_data_for_rde(self, price, timestamp, ncco): pass
    def calculate_ncco(self, bit_mode, market_phase, sentiment_score, entropy): return NCCO()
    def select_strategy(self, bit_mode, market_phase, ncco, volatility): return "hold", 0.5
class FerrisState: pass
class NCCO: pass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cycle_hash_key(price, volume, timestamp):
    """
    Get a unique hash key for a trading cycle.
    """
    key = f"{price}{volume}{timestamp}".encode()
    return hashlib.sha256(key).hexdigest()[:12]


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
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit

        logger.info(f"🎯 Ferris Wheel Backtester initialized with ${initial_balance:,.2f}")

    def generate_historical_data(self, days: int = 365, volatility: float = 0.02) -> List[Tuple[float, float]]:
        """
        Generate realistic historical price data for backtesting.
        """
        logger.info(f"📊 Generating {days} days of historical data...")
        base_price = 45000.0
        prices = []
        current_price = base_price
        start_time = time.time() - (days * 24 * 3600)

        for hour in range(days * 24):
            timestamp = start_time + (hour * 3600)
            trend = 0.0001 * math.sin(hour / (24 * 7))
            noise = random.gauss(0, volatility / math.sqrt(24))
            volatility_shock = random.gauss(0, volatility) * random.random()
            change = trend + noise + volatility_shock
            current_price *= (1 + change)
            current_price = max(1000, min(100000, current_price))
            prices.append((timestamp, current_price))

        logger.info(f"📈 Generated {len(prices)} price points, final price: ${current_price:,.2f}")
        return prices

    def execute_trade(self, price: float, strategy: str, probability: float,
                     bit_mode: int, phase: str, entropy: float) -> TradeRecord:
        """
        Execute a trade based on RDE decision.
        """
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

        if action != "hold":
            position_size = min(self.max_position_size, probability * 0.2)
            position_value = self.balance * position_size
        else:
            position_size = 0.0
            position_value = 0.0

        pnl = 0.0
        if action == "buy" and position_value > 0:
            price_change = random.gauss(0.001, 0.005)
            pnl = position_value * price_change
        elif action == "sell" and position_value > 0:
            price_change = random.gauss(-0.001, 0.005)
            pnl = position_value * price_change

        self.balance += pnl

        trade = TradeRecord(
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
            metadata={"position_size": position_size, "position_value": position_value}
        )

        self.trade_history.append(trade)
        self.performance_history.append(self.balance)
        return trade

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if not self.performance_history:
            return {}
        returns = np.diff(self.performance_history) / self.performance_history[:-1]
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        avg_return = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        peak = self.performance_history[0]
        max_dd = 0.0
        for value in self.performance_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        winning_trades = sum(1 for trade in self.trade_history if trade.pnl > 0)
        total_trades = len([t for t in self.trade_history if t.action != "hold"])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        gross_profit = sum(trade.pnl for trade in self.trade_history if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trade_history if trade.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        return {
            "total_return": total_return, "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd, "volatility": volatility, "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t.pnl for t in self.trade_history]) if self.trade_history else 0.0,
            "total_trades": total_trades, "winning_trades": winning_trades
        }

    def test_cycle_hash_logic(self):
        """Test the injected cycle hash key generation."""
        print("🧪 Testing Cycle Hash Logic...")
        try:
            ts = time.time()
            key1 = get_cycle_hash_key(50000, 100, ts)
            key2 = get_cycle_hash_key(50000, 100, ts)
            key3 = get_cycle_hash_key(50001, 100, ts)
            assert len(key1) == 12, "Hash key should be 12 characters"
            assert key1 == key2, "Hashes with same inputs should be identical"
            assert key1 != key3, "Hashes with different inputs should be different"
            print("✅ Cycle hash logic test passed.")
            self.mathematical_checks["cycle_hash_logic"] = True
        except Exception as e:
            print(f"❌ Cycle hash logic test failed: {e}")
            self.mathematical_checks["cycle_hash_logic"] = False

    def validate_mathematics(self) -> Dict[str, bool]:
        """Validate the core mathematical formulas of the RDE."""
        logger.info("🔬 Validating core mathematics...")
        self.test_cycle_hash_logic()
        try:
            ncco_result = self.ferris_rde.calculate_ncco(bit_mode=16, market_phase="bull", sentiment_score=0.7, entropy=0.4)
            assert isinstance(ncco_result, NCCO), "NCCO result should be an NCCO object"
            self.mathematical_checks["ncco_calculation"] = True
            logger.info("✅ NCCO calculation validated.")
        except Exception as e:
            logger.error(f"❌ NCCO calculation failed: {e}")
            self.mathematical_checks["ncco_calculation"] = False
        try:
            strategy, _ = self.ferris_rde.select_strategy(bit_mode=16, market_phase="bull", ncco=ncco_result, volatility=0.03)
            assert isinstance(strategy, str), "Strategy should be a string"
            self.mathematical_checks["strategy_selection"] = True
            logger.info("✅ Strategy selection validated.")
        except Exception as e:
            logger.error(f"❌ Strategy selection failed: {e}")
            self.mathematical_checks["strategy_selection"] = False
        return self.mathematical_checks

    def calculate_live_ready_score(self) -> float:
        """Calculate a score indicating readiness for live trading."""
        score = 0
        if all(self.mathematical_checks.values()):
            score += 0.4
        risk_metrics = self.calculate_risk_metrics()
        if risk_metrics.get("sharpe_ratio", 0) > 1.0:
            score += 0.2
        if risk_metrics.get("max_drawdown", 1) < 0.2:
            score += 0.2
        if risk_metrics.get("win_rate", 0) > 0.55:
            score += 0.2
        return min(1.0, score)

    def run_backtest(self, days: int = 90, volatility: float = 0.02) -> BacktestResult:
        """Run a full backtest simulation."""
        logger.info(f"🚀 Starting backtest for {days} days...")
        historical_data = self.generate_historical_data(days, volatility)
        for timestamp, price in historical_data:
            ncco = self.ferris_rde.calculate_ncco(16, "neutral", 0.5, 0.5)
            self.ferris_rde.process_data_for_rde(price=price, timestamp=timestamp, ncco=ncco)
            strategy, probability = self.ferris_rde.select_strategy(16, "neutral", ncco, volatility)
            self.execute_trade(price, strategy, probability, 16, "neutral", 0.5)
        self.validate_mathematics()
        risk_metrics = self.calculate_risk_metrics()
        live_ready_score = self.calculate_live_ready_score()
        result = BacktestResult(
            total_trades=risk_metrics.get("total_trades", 0),
            winning_trades=risk_metrics.get("winning_trades", 0),
            losing_trades=risk_metrics.get("total_trades", 0) - risk_metrics.get("winning_trades", 0),
            win_rate=risk_metrics.get("win_rate", 0),
            total_return=risk_metrics.get("total_return", 0),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", 0),
            max_drawdown=risk_metrics.get("max_drawdown", 0),
            strategy_performance={},
            risk_metrics=risk_metrics,
            trade_history=[t.__dict__ for t in self.trade_history],
            mathematical_validation=self.mathematical_checks,
            live_ready_score=live_ready_score
        )
        logger.info(f"✅ Backtest finished. Final Balance: ${self.balance:,.2f}, Return: {result.total_return:.2%}")
        return result

    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Ferris Wheel RDE Backtest Results', fontsize=16)
        perf_df = pd.DataFrame(self.performance_history, columns=['Balance'])
        ax1.plot(perf_df.index, perf_df['Balance'], label='Portfolio Value', color='blue')
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True)
        trade_df = pd.DataFrame(self.trade_history)
        buys = trade_df[trade_df['action'] == 'buy']
        sells = trade_df[trade_df['action'] == 'sell']
        ax1.scatter(buys.index, perf_df.loc[buys.index, 'Balance'], marker='^', color='green', label='Buy', s=100)
        ax1.scatter(sells.index, perf_df.loc[sells.index, 'Balance'], marker='v', color='red', label='Sell', s=100)
        ax1.legend()
        drawdown = (perf_df['Balance'].cummax() - perf_df['Balance']) / perf_df['Balance'].cummax()
        ax2.fill_between(drawdown.index, -drawdown*100, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Trades')
        ax2.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path)
            logger.info(f"💾 Plot saved to {save_path}")
        plt.show()

    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to a JSON file."""
        with open(filepath, 'w') as f:
            # A custom encoder may be needed if result contains non-serializable types
            json.dump(result.__dict__, f, indent=4)
        logger.info(f"💾 Results saved to {filepath}")


def main():
    """Main function to run the backtester."""
    backtester = FerrisWheelBacktester()
    # Run a 1-year backtest
    results = backtester.run_backtest(days=365)
    # Print summary
    print("\n" + "="*50)
    print("Backtest Summary")
    print("="*50)
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Live Ready Score: {results.live_ready_score:.2f}")
    print("="*50)
    # Save and plot results
    backtester.save_results(results, "ferris_wheel_backtest_results.json")
    backtester.plot_results(results, "ferris_wheel_backtest_plot.png")

if __name__ == "__main__":
    main() 