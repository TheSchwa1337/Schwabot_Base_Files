"""
Unified Profit Vectorization System
-----------------------------------
Provides functionalities for calculating, analyzing, and vectorizing profit
metrics across various trading strategies and timeframes.
This system is crucial for performance evaluation and optimization.
"""

import numpy as np
from typing import Dict, List, Any, Union


class UnifiedProfitVectorizationSystem:
    """
    Manages the calculation and vectorization of profit-related metrics.
    """

    def __init__(self):
        """Initializes the profit vectorization system."""
        self.profit_history: List[float] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_profit": 0.0,
            "average_profit_per_trade": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,  # Placeholder
            "sortino_ratio": 0.0  # Placeholder
        }

    def calculate_trade_profit(self, entry_price: float, exit_price: float,
                               quantity: float, trade_direction: str) -> float:
        """
        Calculates the profit or loss for a single trade.

        Args:
            entry_price: The price at which the asset was entered.
            exit_price: The price at which the asset was exited.
            quantity: The quantity of the asset traded.
            trade_direction: 'buy' for long, 'sell' for short.

        Returns:
            The calculated profit or loss.
        """
        if trade_direction.lower() == 'buy':
            profit = (exit_price - entry_price) * quantity
        elif trade_direction.lower() == 'sell':
            profit = (entry_price - exit_price) * quantity
        else:
            raise ValueError("Trade direction must be 'buy' or 'sell'.")

        self.profit_history.append(profit)
        return profit

    def update_performance_metrics(self):
        """
        Updates overall performance metrics based on the profit history.
        This is a simplified update; real calculation would involve more data.
        """
        if not self.profit_history:
            return

        profits = np.array(self.profit_history)

        total_trades = len(profits)
        winning_trades = np.sum(profits > 0)
        losing_trades = np.sum(profits < 0)

        self.performance_metrics["total_profit"] = np.sum(profits)
        self.performance_metrics["average_profit_per_trade"] = np.mean(profits)
        self.performance_metrics["win_rate"] = winning_trades / \
            total_trades if total_trades > 0 else 0.0
        self.performance_metrics["loss_rate"] = losing_trades / \
            total_trades if total_trades > 0 else 0.0

        # Max Drawdown (simple calculation for demonstration)
        cumulative_returns = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        self.performance_metrics["max_drawdown"] = np.max(
            drawdown) if len(drawdown) > 0 else 0.0

        # Sharpe and Sortino Ratios would require more data (e.g., risk-free rate, daily returns)
        # For now, they remain as placeholders or can be calculated with
        # external data.

    def get_profit_vector(self, data: List[float]) -> np.ndarray:
        """
        Converts a list of profit values into a NumPy array (vector).

        Args:
            data: A list of profit values.

        Returns:
            A NumPy array representing the profit vector.
        """
        return np.array(data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Returns the current performance metrics.
        """
        self.update_performance_metrics()  # Ensure metrics are up-to-date
        return self.performance_metrics


if __name__ == "__main__":
    print("--- Unified Profit Vectorization System Demo ---")
    profit_system = UnifiedProfitVectorizationSystem()

    # Simulate some trades
    profits = [
        profit_system.calculate_trade_profit(100, 105, 10, 'buy'),  # +50
        profit_system.calculate_trade_profit(50, 48, 20, 'sell'),  # +40
        profit_system.calculate_trade_profit(200, 190, 5, 'buy'),   # -50
        profit_system.calculate_trade_profit(10, 12, 100, 'buy')  # +200
    ]

    print(f"Individual Trade Profits: {profits}")

    # Get performance summary
    summary = profit_system.get_performance_summary()
    print("\nPerformance Summary:")
    for k, v in summary.items():
        if isinstance(v, (float, np.float64)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Get a profit vector
    profit_vector = profit_system.get_profit_vector(
        profit_system.profit_history)
    print(f"\nProfit Vector: {profit_vector}")
