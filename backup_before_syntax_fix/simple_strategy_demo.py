import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import numpy as np

#!/usr/bin/env python3
"""
🧠 Schwabot Mathematical Strategy Demo
=====================================

Simplified demo showcasing the core mathematical implementations:
- Real Sharpe/Sortino ratio calculations
- Kelly criterion position sizing
- MCMC profit state modeling
- Real volatility calculations
- FlipSwitch logic integration

This demonstrates 31 days of mathematical framework development
in a self-contained, flake8-compliant script.
"""




class PerformanceMetrics:
    """Real Sharpe/Sortino ratio calculations."""

    def __init__(self, risk_free_rate: float = 0.2):
        self.risk_free_rate = risk_free_rate
        self.returns_history: List[float] = []

    def add_return(self, return_pct: float):
        """Add a return to the history."""
        self.returns_history.append(return_pct)
        if len(self.returns_history) > 252:  # Keep last year
            self.returns_history = self.returns_history[-252:]

    def calculate_sharpe_ratio():-> float:
        """Calculate real Sharpe ratio."""
        if len(self.returns_history) < 10:
            return 0.0

        returns_array = np.array(self.returns_history)
        excess_returns = returns_array - (self.risk_free_rate / 252)

        std_dev = np.std(excess_returns, ddof=1)
        if std_dev == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / std_dev * np.sqrt(252)
        return float(sharpe)

    def calculate_sortino_ratio():-> float:
        """Calculate real Sortino ratio (downside deviation, only)."""
        if len(self.returns_history) < 10:
            return 0.0

        returns_array = np.array(self.returns_history)
        excess_returns = returns_array - (self.risk_free_rate / 252)

        # Only negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0"

        downside_deviation = np.std(downside_returns, ddof=1)
        if downside_deviation == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        return float(sortino)


class KellyCriterion:
    """Kelly criterion for optimal position sizing."""

    def __init__(self):
        self.trade_history: List[Dict[str, float]] = []

    def add_trade_result(self, profit: float, loss: float, won: bool):
        """Add trade result for Kelly calculation."""
        self.trade_history.append()
            {"profit": profit if won else 0, "loss": loss if not won else 0, "won": won}
        )

        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def calculate_kelly_fraction():-> float:
        """Calculate Kelly criterion fraction."""
        if len(self.trade_history) < 10:
            return 0.25  # Conservative default

        wins = [t for t in self.trade_history if t["won"]]
        losses = [t for t in self.trade_history if not t["won"]]

        if len(wins) == 0 or len(losses) == 0:
            return 0.25

        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([w["profit"] for w in wins])
        avg_loss = np.mean([l["loss"] for l in losses])

        if avg_loss == 0:
            return 0.25

        # Kelly formula: f = (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = 1-p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Conservative scaling (max 25% of, capital)
        return max(0.5, min(0.25, kelly_fraction))


class MarkovProfitPredictor:
    """MCMC profit state prediction system."""

    def __init__(self):
        self.states = ["loss", "neutral", "small_profit", "big_profit"]
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.current_state = "neutral"
        self.state_history = deque(maxlen=50)

    def classify_profit():-> str:
        """Classify profit into discrete states."""
        if profit_pct < -0.2:
            return "loss"
        elif profit_pct < 0.05:
            return "neutral"
        elif profit_pct < 0.3:
            return "small_profit"
        else:
            return "big_profit"

    def update_state(self, profit_pct: float):
        """Update Markov chain with new profit data."""
        new_state = self.classify_profit(profit_pct)

        # Update transition matrix
        self.transition_matrix[self.current_state][new_state] += 1
        self.state_counts[self.current_state] += 1

        # Update current state and history
        self.state_history.append(new_state)
        self.current_state = new_state

    def predict_next_state():-> Tuple[str, float]:
        """Predict next state with probability."""
        if self.state_counts[self.current_state] == 0:
            return "neutral", 0.25

        transitions = self.transition_matrix[self.current_state]
        total = self.state_counts[self.current_state]

        # Find most likely next state
        best_state = self.current_state
        best_prob = 0.0

        for state, count in transitions.items():
            prob = count / total
            if prob > best_prob:
                best_prob = prob
                best_state = state

        return best_state, best_prob


class VolatilityCalculator:
    """Real volatility calculation system."""

    def __init__(self, window: int = 20):
        self.window = window
        self.price_history: List[float] = []

    def add_price(self, price: float):
        """Add price to history."""
        self.price_history.append(price)
        if len(self.price_history) > self.window * 2:
            self.price_history = self.price_history[-self.window * 2 :]

    def calculate_volatility():-> float:
        """Calculate rolling volatility."""
        if len(self.price_history) < 2:
            return 0.2  # Default 2%

        # Calculate returns
        returns = []
        for i in range(1, len(self.price_history)):
            if self.price_history[i - 1] != 0:
                ret = ()
                    self.price_history[i] - self.price_history[i - 1]
                ) / self.price_history[i - 1]
                returns.append(ret)

        if len(returns) < 2:
            return 0.2

        # Use recent window
        recent_returns = ()
            returns[-self.window :] if len(returns) > self.window else returns
        )
        volatility = np.std(recent_returns, ddof=1)

        # Annualize (assuming daily, data)
        return float(volatility * np.sqrt(252))


class FlipSwitchLogic:
    """FlipSwitch trading logic with Kelly integration."""

    def __init__(self):
        self.aggressive_mode = False
        self.switch_threshold = 0.6

    def should_flip_aggressive():-> Tuple[bool, float]:
        """
        Determine if strategy should flip to aggressive mode.

        Returns:
            (should_flip, confidence_score)
        """
        confidence = 0.5

        # More liberal conditions for demo purposes
        # High Kelly + reasonable volatility + any momentum = aggressive
        if ()
            kelly_fraction > 0.1
            and volatility < 0.8
            and abs(momentum) > 0.05
            and forecast_confidence > 0.2
        ):
            self.aggressive_mode = True
            confidence = min(0.8, kelly_fraction + forecast_confidence + 0.2)

        # Only extreme conditions = conservative
        elif volatility > 1.0 or forecast_confidence < 0.1:
            self.aggressive_mode = False
            confidence = 0.2

        # Default to moderate confidence for trade execution
        else:
            confidence = 0.5

        return self.aggressive_mode, confidence


class IntegratedStrategy:
    """Integrated strategy combining all mathematical frameworks."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Initialize subsystems
        self.performance = PerformanceMetrics()
        self.kelly = KellyCriterion()
        self.markov = MarkovProfitPredictor()
        self.volatility = VolatilityCalculator()
        self.flipswitch = FlipSwitchLogic()

        # Trading state
        self.trades_executed = 0
        self.winning_trades = 0
        self.portfolio_history = [initial_capital]

    def execute_trading_cycle():-> Dict[str, Any]:
        """Execute one complete trading cycle."""
        # 1. Update price and calculate volatility
        self.volatility.add_price(current_price)
        current_volatility = self.volatility.calculate_volatility()

        # 2. Calculate momentum (simplified)
        momentum = 0.0
        if len(self.volatility.price_history) >= 5:
            recent_prices = self.volatility.price_history[-5:]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # 3. Get Kelly position sizing
        kelly_fraction = self.kelly.calculate_kelly_fraction()

        # 4. Get Markov forecast
        forecast_state, forecast_confidence = self.markov.predict_next_state()

        # 5. FlipSwitch decision
        aggressive_mode, strategy_confidence = self.flipswitch.should_flip_aggressive()
            kelly_fraction, current_volatility, momentum, forecast_confidence
        )

        # 6. Calculate position size
        base_position = 0.1  # 10% of capital
        if aggressive_mode:
            position_multiplier = kelly_fraction * 1.5
        else:
            position_multiplier = kelly_fraction * 0.8

        position_size = min(base_position * position_multiplier, 0.25)  # Max 25%

        # 7. Simulate trade execution
        trade_executed = strategy_confidence > 0.3  # Lowered threshold for demo
        profit_pct = 0.0

        if trade_executed:
            # Simulate trade outcome based on confidence
            win_probability = strategy_confidence * 0.8  # Conservative adjustment

            if random.random() < win_probability:
                # Winning trade
                profit_pct = random.uniform(0.1, 0.5) * position_size
                self.winning_trades += 1
                won = True
            else:
                # Losing trade
                profit_pct = random.uniform(-0.5, -0.1) * position_size
                won = False

            # Update capital
            self.current_capital *= 1 + profit_pct
            self.portfolio_history.append(self.current_capital)

            # Update systems with trade result
            self.performance.add_return(profit_pct)
            self.kelly.add_trade_result()
                profit=profit_pct if won else 0,
                loss=abs(profit_pct) if not won else 0,
                won=won,
            )
            self.markov.update_state(profit_pct)
            self.trades_executed += 1

        # 8. Calculate performance metrics
        sharpe = self.performance.calculate_sharpe_ratio()
        sortino = self.performance.calculate_sortino_ratio()
        total_return = ()
            self.current_capital - self.initial_capital
        ) / self.initial_capital
        win_rate = self.winning_trades / max(1, self.trades_executed)

        return {}
            "price": current_price,
            "portfolio_value": self.current_capital,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "kelly_fraction": kelly_fraction,
            "volatility": current_volatility,
            "momentum": momentum,
            "forecast_state": forecast_state,
            "forecast_confidence": forecast_confidence,
            "aggressive_mode": aggressive_mode,
            "strategy_confidence": strategy_confidence,
            "position_size": position_size,
            "trade_executed": trade_executed,
            "profit_pct": profit_pct,
            "trades_executed": self.trades_executed,
            "win_rate": win_rate,
        }


def main():
    """Run the integrated strategy demo."""
    print("🧠 Schwabot Mathematical Strategy Demo")
    print("=" * 50)
    print("Showcasing 31 days of mathematical framework development:")
    print("✅ Real Sharpe/Sortino calculations")
    print("✅ Kelly criterion position sizing")
    print("✅ MCMC profit state prediction")
    print("✅ Real volatility calculations")
    print("✅ FlipSwitch-Kelly integration")
    print("=" * 50)

    # Initialize strategy
    strategy = IntegratedStrategy(initial_capital=100000.0)

    # Run simulation
    print("\n🎯 Running 50 Trading Cycles...")
    print()
        f"{'Cycle':<5} {'Price':<8} {'Portfolio':<12} {'Return':<8} {'Sharpe':<8} "
        f"{'Kelly':<7} {'Mode':<6} {'Trades':<6}"
    )
    print("-" * 70)

    base_price = 50000.0
    for cycle in range(1, 51):
        # Simulate realistic price movement
        if cycle == 1:
            price = base_price
        else:
            # Add some realistic volatility and momentum
            volatility = random.uniform(0.05, 0.3)
            momentum = random.uniform(-0.1, 0.1)
            price_change = base_price * (momentum + np.random.normal(0, volatility))
            price = max(base_price + price_change, base_price * 0.5)
            base_price = price

        # Execute trading cycle
        results = strategy.execute_trading_cycle(price)

        # Print results every 10 cycles
        if cycle % 10 == 0:
            mode_str = "AGG" if results["aggressive_mode"] else "CON"
            print()
                f"{cycle:<5} {results['price']:<8.0f} "
                f"${results['portfolio_value']:<11,.0f} "
                f"{results['total_return']:<7.1%} "
                f"{results['sharpe_ratio']:<7.2f} "
                f"{results['kelly_fraction']:<6.2f} "
                f"{mode_str:<6} "
                f"{results['trades_executed']:<6}"
            )

    # Final summary
    final_results = strategy.execute_trading_cycle(price)

    print("\n" + "=" * 70)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"💰 Initial Capital:      ${strategy.initial_capital:,.2f}")
    print(f"💰 Final Capital:        ${final_results['portfolio_value']:,.2f}")
    print(f"📈 Total Return:         {final_results['total_return']:.2%}")
    print(f"📊 Sharpe Ratio:         {final_results['sharpe_ratio']:.4f}")
    print(f"📊 Sortino Ratio:        {final_results['sortino_ratio']:.4f}")
    print(f"🎯 Total Trades:         {final_results['trades_executed']}")
    print(f"🏆 Win Rate:             {final_results['win_rate']:.1%}")
    print(f"🎲 Kelly Fraction:       {final_results['kelly_fraction']:.3f}")
    print(f"📊 Current Volatility:   {final_results['volatility']:.1%}")
    print()
        f"🔄 Strategy Mode:        "
        f"{'Aggressive' if final_results['aggressive_mode'] else 'Conservative'}"
    )

    print("\n🎉 Demo Complete!")
    print("This demonstrates the integration of advanced mathematical frameworks")
    print("developed over 31 days of systematic enhancement to Schwabot's core.")'


if __name__ == "__main__":
    main()
