from typing import Any, Dict, List, Optional

import numpy as np




"""
Unified Profit Vectorization System
-----------------------------------
Provides functionalities for calculating, analyzing, and vectorizing profit
metrics across various trading strategies and timeframes.
This system is crucial for performance evaluation and optimization."
"""

class UnifiedProfitVectorizationSystem:
    """
    Manages the calculation and vectorization of profit-related metrics for mathematical pipeline.
"""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initializes the profit vectorization system for tensor bucket operations."""
        self.profit_history: List[float] = []
        self.risk_free_rate = risk_free_rate
        self.performance_metrics: Dict[str, Any] = {
            "total_profit": 0.0,
            "average_profit_per_trade": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
}

    def calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted returns in tick analysis.

        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (defaults to instance rate)

        Returns:
            Sharpe ratio value for probabilistic drive systems
        """
        if not returns or len(returns) < 2:
            return 0.0

        risk_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_rate

        std_dev = np.std(excess_returns, ddof=1)
        if std_dev == 0 or np.isnan(std_dev):
            return 0.0

        sharpe = np.mean(excess_returns) / std_dev
        return float(sharpe)

    def calculate_sortino_ratio(
        self, returns: List[float], risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sortino ratio focusing on downside deviation for jerf pattern analysis.

        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (defaults to instance rate)

        Returns:
            Sortino ratio value for probabilistic drive systems
        """
        if not returns or len(returns) < 2:
            return 0.0

        risk_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_rate

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns, ddof=1)
        if downside_deviation == 0 or np.isnan(downside_deviation):
            return 0.0

        sortino = np.mean(excess_returns) / downside_deviation
        return float(sortino)

    def calculate_kelly_criterion(
self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing in tick analysis.

Args:
            win_rate: Probability of winning (0-1)
avg_win: Average winning amount
avg_loss: Average losing amount (positive value)

Returns:
            Kelly fraction (0-1) for mathematical pipeline optimization
"""
if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
            return 0.0

loss_rate = 1 - win_rate
reward_risk_ratio = avg_win / avg_loss

kelly_fraction = (reward_risk_ratio * win_rate - loss_rate) / reward_risk_ratio

# Clamp between 0 and 1 for safety
        return max(0.0, min(1.0, kelly_fraction))

def calculate_trade_profit(
        self,
entry_price: float,
exit_price: float,
quantity: float,
trade_direction: str,
    ) -> float:
        """
        Calculates the profit or loss for a single trade in the pipeline.

Args:
            entry_price: The price at which the asset was entered.
exit_price: The price at which the asset was exited.
            quantity: The quantity of the asset traded.
trade_direction: 'buy' for long, 'sell' for short.

Returns:
            The calculated profit or loss for tensor bucket operations.
        """
if trade_direction.lower() == "buy":
            profit = (exit_price - entry_price) * quantity
        elif trade_direction.lower() == "sell":
            profit = (entry_price - exit_price) * quantity
        else:
            raise ValueError("Trade direction must be 'buy' or 'sell'.")

self.profit_history.append(profit)
        return profit

    def calculate_returns_from_profits(self, initial_capital: float = 10000.0) -> List[float]:
        """
        Convert profit history to returns for ratio calculations in mathematical pipeline.

Args:
            initial_capital: Starting capital amount

Returns:
            List of return percentages for probabilistic drive analysis
"""
if not self.profit_history:
            return []

returns = []
capital = initial_capital

for profit in self.profit_history:
            if capital > 0:
                return_pct = profit / capital
returns.append(return_pct)
capital += profit
else:
                returns.append(0.0)

        return returns

    def update_performance_metrics(self, initial_capital: float = 10000.0):
        """
        Updates overall performance metrics based on the profit history for mathematical confirmations.
"""
if not self.profit_history:
            return

profits = np.array(self.profit_history)
        returns = self.calculate_returns_from_profits(initial_capital)

total_trades = len(profits)
        winning_trades = np.sum(profits > 0)
        losing_trades = np.sum(profits < 0)

        self.performance_metrics["total_profit"] = float(np.sum(profits))
        self.performance_metrics["average_profit_per_trade"] = float(np.mean(profits))
self.performance_metrics["win_rate"] = (
winning_trades / total_trades if total_trades > 0 else 0.0
        )
self.performance_metrics["loss_rate"] = (
losing_trades / total_trades if total_trades > 0 else 0.0
)

        # Max Drawdown calculation for jerf pattern waveform analysis
cumulative_returns = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        self.performance_metrics["max_drawdown"] = float(max_drawdown)

        # Calculate Sharpe and Sortino ratios for tensor bucket optimization
        if len(returns) > 1:
            self.performance_metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
self.performance_metrics["sortino_ratio"] = self.calculate_sortino_ratio(returns)

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss) for mathematical pipeline.

Returns:
            Profit factor for probabilistic drive systems
        """
        if not self.profit_history:
            return 0.0

profits = np.array(self.profit_history)
        gross_profit = np.sum(profits[profits > 0])
        gross_loss = abs(np.sum(profits[profits < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for mathematical confirmations.
        
        Returns:
            Dictionary containing all performance metrics for tensor bucket analysis
        """
        return {
            **self.performance_metrics,
            "profit_factor": self.calculate_profit_factor(),
            "total_trades": len(self.profit_history),
            "risk_free_rate": self.risk_free_rate,
        }

    def vectorize_profit_patterns(self) -> Dict[str, Any]:
        """
        Vectorize profit patterns for jerf pattern waveform analysis.
        
        Returns:
            Vectorized profit data for mathematical pipeline integration
        """
        if not self.profit_history:
            return {"error": "No profit history available"}

        profits = np.array(self.profit_history)
        
        return {
            "profit_vector": profits.tolist(),
            "profit_magnitude": float(np.linalg.norm(profits)),
            "profit_mean": float(np.mean(profits)),
            "profit_std": float(np.std(profits)),
            "profit_correlation": self._calculate_autocorrelation(profits),
            "profit_trend": self._calculate_trend(profits),
        }

    def _calculate_autocorrelation(self, data: np.ndarray) -> float:
        """Calculate autocorrelation for pattern analysis."""
        if len(data) < 2:
            return 0.0

        # Simple lag-1 autocorrelation
        return float(np.corrcoef(data[:-1], data[1:])[0, 1]) if len(data) > 1 else 0.0

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend slope for mathematical pipeline."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return float(slope)

# Global instance for mathematical pipeline integration
profit_vectorization_system = UnifiedProfitVectorizationSystem()

__all__ = ["UnifiedProfitVectorizationSystem", "profit_vectorization_system"]

if __name__ == "__main__":
    print("--- Unified Profit Vectorization System Demo ---")
    profit_system = UnifiedProfitVectorizationSystem()

# Simulate some trades
    profits = [
        profit_system.calculate_trade_profit(100, 105, 10, "buy"),  # +50
        profit_system.calculate_trade_profit(50, 48, 20, "sell"),  # +40
        profit_system.calculate_trade_profit(200, 190, 5, "buy"),  # -50
        profit_system.calculate_trade_profit(10, 12, 100, "buy"),  # +200
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
    profit_vector = profit_system.get_profit_vector(profit_system.profit_history)
    print(f"\nProfit Vector: {profit_vector}")
    print(f"Kelly Position Multiplier: {profit_system.get_kelly_position_size():.4f}")