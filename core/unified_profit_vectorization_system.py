"""
Unified Profit Vectorization System
-----------------------------------
Provides functionalities for calculating, analyzing, and vectorizing profit
metrics across various trading strategies and timeframes.
This system is crucial for performance evaluation and optimization.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class UnifiedProfitVectorizationSystem:
    """
    Manages the calculation and vectorization of profit-related metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initializes the profit vectorization system."""
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
        Calculate Sharpe ratio for risk-adjusted returns.
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (defaults to instance rate)
            
        Returns:
            Sharpe ratio value
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
        Calculate Sortino ratio focusing on downside deviation.
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (defaults to instance rate)
            
        Returns:
            Sortino ratio value
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
        Calculate Kelly Criterion for optimal position sizing.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)
            
        Returns:
            Kelly fraction (0-1)
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
        Calculates the profit or loss for a single trade.

        Args:
            entry_price: The price at which the asset was entered.
            exit_price: The price at which the asset was exited.
            quantity: The quantity of the asset traded.
            trade_direction: 'buy' for long, 'sell' for short.

        Returns:
            The calculated profit or loss.
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
        Convert profit history to returns for ratio calculations.
        
        Args:
            initial_capital: Starting capital amount
            
        Returns:
            List of return percentages
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
        Updates overall performance metrics based on the profit history.
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

        # Max Drawdown calculation
        cumulative_returns = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / np.maximum(peak, 1.0)
        self.performance_metrics["max_drawdown"] = (
            float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        )

        # Calculate real Sharpe and Sortino ratios
        self.performance_metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
        self.performance_metrics["sortino_ratio"] = self.calculate_sortino_ratio(returns)

    def get_kelly_position_size(
        self, base_position_size: float = 1.0
    ) -> float:
        """
        Get Kelly-optimized position size based on historical performance.
        
        Args:
            base_position_size: Base position size to scale
            
        Returns:
            Kelly-adjusted position size
        """
        if len(self.profit_history) < 10:  # Need minimum history
            return base_position_size * 0.5  # Conservative until we have data
            
        profits = np.array(self.profit_history)
        wins = profits[profits > 0]
        losses = np.abs(profits[profits < 0])
        
        if len(wins) == 0 or len(losses) == 0:
            return base_position_size * 0.5
            
        win_rate = len(wins) / len(profits)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses))
        
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Apply conservative scaling (never risk more than 25% of capital)
        scaled_kelly = min(kelly_fraction, 0.25)
        
        return base_position_size * scaled_kelly

    def get_profit_vector(self, data: List[float]) -> np.ndarray:
        """
        Converts a list of profit values into a NumPy array (vector).

        Args:
            data: A list of profit values.

        Returns:
            A NumPy array representing the profit vector.
        """
        return np.array(data)

    def get_performance_summary(self, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Returns the current performance metrics.
        """
        self.update_performance_metrics(initial_capital)
        
        # Add additional metrics
        summary = self.performance_metrics.copy()
        
        if self.profit_history:
            summary["kelly_position_multiplier"] = self.get_kelly_position_size()
            summary["total_trades"] = len(self.profit_history)
            summary["profit_factor"] = self._calculate_profit_factor()
            summary["calmar_ratio"] = self._calculate_calmar_ratio()
            
        return summary

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.profit_history:
            return 0.0
            
        profits = np.array(self.profit_history)
        gross_profit = np.sum(profits[profits > 0])
        gross_loss = np.abs(np.sum(profits[profits < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return float(gross_profit / gross_loss)

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if self.performance_metrics["max_drawdown"] == 0:
            return 0.0
            
        annual_return = self.performance_metrics["average_profit_per_trade"] * 252  # Assuming daily trades
        return float(annual_return / self.performance_metrics["max_drawdown"])


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
