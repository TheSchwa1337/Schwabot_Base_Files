import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

"""
Ferris Wheel RDE Isolated Backtesting System
============================================

Comprehensive backtesting framework for the Ferris Wheel RDE system.
Tests mathematical logic, strategy performance, risk metrics, and provides
validation for live trading readiness.

This version is completely isolated - no external imports from core module.
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Ferris Wheel RDE Implementation (Isolated) ---


@dataclass
    class FerrisState:
    """Represents a Ferris Wheel state."""

    phase: int
    bit_state: int
    rotation_count: int
    entropy_level: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
    class NCCOHistoryRecord:
    """Represents a feedback record for NCCO memory."""

    timestamp: float
    strategy: str
    weight: float
    phase_modifier: float
    probability: float
    phase: str
    meta: Dict[str, Any] = field(default_factory=dict)


class FerrisWheelRDE:
    """
    Ferris Wheel RDE (Recursive Dualistic, Engine) implementation.
    Isolated version for backtesting.
    """

    def __init__(self, max_phases: int = 256):
        self.max_phases = max_phases
        self.current_phase = 0
        self.rotation_count = 0
        self.bit_phase = 4  # Default to 4-bit
        self.states: List[FerrisState] = []
        self.memory_bank: Dict[str, Any] = {}
        self.ncco_history: List[NCCOHistoryRecord] = []
        self.standard_bits = [4, 8, 42]
        self.alpha = 0.3  # Smoothing factor
        self.learning_rate = 0.5  # Strategy mutation learning rate
        self.reward_rate = 0.1  # Reinforcement learning update rate

        # Strategy configuration
        self.strategy_sets = {4: ["hold"], 8: ["stable_swap"], 42: ["flip", "exit"]}
        self.phase_modifiers = {}
            ("hold", "ascent"): 1.2,
            ("flip", "descent"): 1.3,
            ("exit", "peak"): 1.5,
            ("entry", "trough"): 1.4,
        }
        # Strategy performance tracking with history
        self.strategy_performance: Dict[str, float] = {}
            s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]
        }
        self.strategy_weights: Dict[str, float] = {}
            s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]
        }
        self.strategy_smoothed: Dict[str, float] = {}
            s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]
        }
        self.strategy_performance_history: Dict[str, List[float]] = {}
            s: [] for s in ["hold", "stable_swap", "flip", "exit", "entry"]
        }

        # Multi-timeframe phase analysis
        self.phase_windows = [16, 32, 64]
        self.phase_weights = [0.2, 0.5, 0.3]

        # State tracking
        self.last_phase_angle = 0.0
        self.last_price = None
        self.last_time = None
        self.price_history: List[Tuple[float, float]] = []  # (timestamp, price)

        logger.info(f"🎡 Ferris Wheel RDE initialized with max_phases={max_phases}")

    def _calculate_entropy():-> float:
        """Calculate entropy of a string using Shannon's formula."""'
        if not data:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate probabilities and entropy
        length = len(data)
        entropy = 0.0

        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def normalize_weights():-> Dict[str, float]:
        total = sum(performance.values())
        if total == 0:
            n = len(performance)
            return {k: 1.0 / n for k in performance}
        return {k: v / total for k, v in performance.items()}

    def smooth_weights():-> Dict[str, float]:
        return {}
            k: (1 - alpha) * old_weights.get(k, 1.0) + alpha * performance.get(k, 1.0)
            for k in performance
        }

    def mutate_strategy_weights():-> None:
        """Mutate strategy weights based on success trends."""
        if learning_rate is None:
            learning_rate = self.learning_rate

        for strat, score in success.items():
            if strat not in self.strategy_performance_history:
                self.strategy_performance_history[strat] = []

            hist = self.strategy_performance_history[strat]
            hist.append(score)

            # Keep last 10 performance scores
            if len(hist) > 10:
                hist.pop(0)

            # Calculate trend using linear regression
            if len(hist) > 2:
                x = np.arange(len(hist))
                trend = np.polyfit(x, hist, 1)[0]  # Slope of linear fit
            else:
                trend = 0.0

            # Apply mutation: positive trend increases weight
            delta = learning_rate * trend
            current_perf = self.strategy_performance.get(strat, 1.0)
            self.strategy_performance[strat] = max()
                0.1, current_perf + delta
            )  # Ensure positive

    def update_strategy_reward():-> None:
        """Update strategy performance based on reward/penalty."""
        if reward_rate is None:
            reward_rate = self.reward_rate

        current = self.strategy_performance.get(strategy, 1.0)
        new_performance = (1 - reward_rate) * current + reward_rate * reward

        # Ensure performance stays in reasonable bounds
        self.strategy_performance[strategy] = max(0.1, min(10.0, new_performance))

    def get_strategy_set():-> List[str]:
        return self.strategy_sets.get(bit_mode, ["hold"])

    def update_price_history(self, price: float, timestamp: Optional[float] = None):
        t = timestamp if timestamp is not None else time.time()
        self.price_history.append((t, price))
        if len(self.price_history) > 1000:
            self.price_history.pop(0)

    def get_phase_angle():-> float:
        """Get phase angle using simple price delta method."""
        if len(self.price_history) < 2:
            return 0.0

        # Simple phase estimation using price changes
        if len(self.price_history) >= window:
            prices = [p for _, p in self.price_history[-window:]]
            # Calculate simple phase based on price trend
            if len(prices) >= 2:
                price_change = (prices[-1] - prices[0]) / prices[0]
                # Map to phase angle (0 to 2π)
                phase_angle = (price_change + 1) * math.pi  # Simple mapping
                return phase_angle % (2 * math.pi)
        return 0.0

    def get_phase_state():-> str:
        self.update_price_history(price, timestamp)
        phase_angle = self.get_phase_angle()
        if len(self.price_history) < 2:
            return "ascent"
        # Calculate dP/dt
        (t1, p1), (t0, p0) = self.price_history[-1], self.price_history[-2]
        dpdt = (p1 - p0) / (t1 - t0) if t1 != t0 else 0.0
        # Phase logic
        if dpdt > 0 and 0 <= phase_angle < math.pi:
            return "ascent"
        elif abs(phase_angle - math.pi) < 0.1:
            return "peak"
        elif dpdt < 0 and math.pi <= phase_angle < 2 * math.pi:
            return "descent"
        elif abs(phase_angle - 2 * math.pi) < 0.1:
            return "trough"
        else:
            return "ascent"

    def apply_phase_modifiers():-> Dict[str, float]:
        adjusted = {}
        for s, w in weights.items():
            phi = self.phase_modifiers.get((s, phase), 1.0)
            adjusted[s] = w * phi
        # Re-normalize
        total = sum(adjusted.values())
        if total == 0:
            n = len(adjusted)
            return {k: 1.0 / n for k in adjusted}
        return {k: v / total for k, v in adjusted.items()}

    def softmax():-> Dict[str, float]:
        exp_w = {k: math.exp(v) for k, v in weights.items()}
        total = sum(exp_w.values())
        return {k: v / total for k, v in exp_w.items()}

    def select_strategy():-> Tuple[str, float]:
        if method == "softmax":
            probs = self.softmax(weights)
            strategy = max(probs, key=probs.get)
            return strategy, probs[strategy]
        else:
            strategy = max(weights, key=weights.get)
            total = sum(weights.values())
            prob = weights[strategy] / total if total > 0 else 1.0 / len(weights)
            return strategy, prob

    def ferris_rde_cycle():-> Dict[str, Any]:
        # 1. Update phase state
        phase = self.get_phase_state(price, timestamp)
        # 2. Normalize and smooth strategy weights
        strat_set = self.get_strategy_set(bit_mode)
        perf = {s: self.strategy_performance.get(s, 1.0) for s in strat_set}
        norm = self.normalize_weights(perf)
        smoothed = self.smooth_weights(self.strategy_smoothed, norm, self.alpha)
        # 3. Apply phase modifiers
        adjusted = self.apply_phase_modifiers(smoothed, phase)
        # 4. Select strategy
        strategy, prob = self.select_strategy(adjusted, method="softmax")

        # 5. Generate entropy from strategy decision
        hash_input = f"{strategy}_{bit_mode}_{phase}_{prob:.4f}_{self.current_phase}"
        entropy_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        entropy_val = self._calculate_entropy(entropy_hash)

        # 6. Feedback: update memory
        self.strategy_smoothed.update(smoothed)
        self.strategy_weights.update(norm)
        self.ncco_history.append()
            NCCOHistoryRecord()
                timestamp=timestamp if timestamp is not None else time.time(),
                strategy=strategy,
                weight=smoothed[strategy],
                phase_modifier=self.phase_modifiers.get((strategy, phase), 1.0),
                probability=prob,
                phase=phase,
                meta={"bit_mode": bit_mode, "entropy": entropy_val},
            )
        )

        # 7. Update Ferris state with actual entropy
        self.current_phase = (self.current_phase + 1) % self.max_phases
        self.rotation_count += 1
        ferris_state = FerrisState()
            phase=self.current_phase,
            bit_state=bit_mode,
            rotation_count=self.rotation_count,
            entropy_level=entropy_val,
            metadata={}
                "strategy": strategy,
                "probability": prob,
                "phase": phase,
                "weights": adjusted,
                "hash_input": hash_input,
                "entropy_hash": entropy_hash[:16],  # Store first 16 chars for debugging
            },
        )
        self.states.append(ferris_state)

        # 8. Return decision with RDE context
        return {}
            "strategy": strategy,
            "probability": prob,
            "phase": phase,
            "weights": adjusted,
            "ferris_state": ferris_state,
            "ncco_history": self.ncco_history[-10:],  # last 10
            "rde_context": ferris_state,
        }


# --- Backtesting System ---


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
        """Generate realistic historical price data for backtesting."""
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
        """Execute a trade based on RDE decision."""
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
        """Run comprehensive backtest."""
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
            "mathematical_validation": {}
                k: str(v) for k, v in result.mathematical_validation.items()
            },  # Convert bool to str
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
