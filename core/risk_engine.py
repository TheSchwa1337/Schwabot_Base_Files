"""
Risk Engine
Handles position sizing, risk management, and profit-to-risk expectancy calculations.
"""

import logging
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSignal:
    """Position sizing signal with risk parameters"""
    action: str  # 'entry', 'exit', 'hold'
    base_size: float
    risk_adjusted_size: float
    stop_loss: float
    take_profit: float
    max_loss: float
    expected_return: float
    kelly_fraction: float
    confidence: float
    timestamp: float


@dataclass
class RiskMetrics:
    """Risk metrics and statistics"""
    current_expectancy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    kelly_optimal: float
    current_volatility: float
    var_95: float  # Value at Risk 95%


class RiskEngine:
    """
    Handles risk management, position sizing, and expectancy calculations.
    
    Implements:
    1. Volatility-weighted position sizing using fractional Kelly
    2. Rolling expectancy calculation with Sharpe ratio
    3. Dynamic risk adjustment based on recent performance
    4. VaR (Value at Risk) calculation
    """
    
    def __init__(self, max_position_size: float = 0.25, lookback_period: int = 100):
        """
        Initialize the risk engine.
        
        Args:
            max_position_size: Maximum position size as fraction of capital
            lookback_period: Number of trades to look back for expectancy calculation
        """
        self.max_position_size = max_position_size
        self.lookback_period = lookback_period
        
        # Trade history for expectancy calculation
        self.trade_history: deque = deque(maxlen=lookback_period)
        self.return_history: deque = deque(maxlen=lookback_period * 2)
        
        # Volatility tracking
        self.price_history: deque = deque(maxlen=1000)
        self.volatility_window = 20  # Rolling volatility window
        
        # Risk parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.min_kelly_fraction = 0.01  # Minimum Kelly fraction
        self.max_kelly_fraction = 0.20  # Maximum Kelly fraction
        
        # Performance tracking
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        
        logger.info("Risk engine initialized")

    def calculate_position_size(self, signal_confidence: float, expected_edge: float, 
                              current_price: float, stop_loss_price: float) -> PositionSignal:
        """
        Calculate optimal position size using Kelly criterion with risk adjustments.
        
        Args:
            signal_confidence: Pattern confidence (0-1)
            expected_edge: Expected return per unit risk
            current_price: Current market price
            stop_loss_price: Stop loss price
            
        Returns:
            PositionSignal with risk-adjusted sizing
        """
        # Calculate position risk
        risk_per_unit = abs(current_price - stop_loss_price) / current_price
        
        # Get current volatility
        current_volatility = self._calculate_current_volatility()
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(expected_edge, current_volatility)
        
        # Adjust Kelly fraction based on confidence and recent performance
        confidence_adjustment = signal_confidence * 0.5 + 0.5  # Scale to 0.5-1.0
        performance_adjustment = self._get_performance_adjustment()
        
        adjusted_kelly = kelly_fraction * confidence_adjustment * performance_adjustment
        adjusted_kelly = np.clip(adjusted_kelly, self.min_kelly_fraction, self.max_kelly_fraction)
        
        # Calculate position size
        base_size = adjusted_kelly / risk_per_unit if risk_per_unit > 0 else 0
        risk_adjusted_size = min(base_size, self.max_position_size)
        
        # Additional risk checks
        if self.current_drawdown > self.max_drawdown_limit * 0.8:
            risk_adjusted_size *= 0.5  # Reduce size during drawdown
            
        # Calculate take profit (risk-reward ratio)
        risk_reward_ratio = 2.0  # Target 2:1 reward-to-risk
        take_profit_price = current_price + (risk_per_unit * current_price * risk_reward_ratio)
        
        # Calculate expected return
        win_rate = self._calculate_win_rate()
        expected_return = (win_rate * risk_reward_ratio - (1 - win_rate)) * risk_per_unit
        
        signal = PositionSignal(
            action='entry',
            base_size=base_size,
            risk_adjusted_size=risk_adjusted_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            max_loss=risk_adjusted_size * risk_per_unit,
            expected_return=expected_return,
            kelly_fraction=adjusted_kelly,
            confidence=signal_confidence,
            timestamp=time.time()
        )
        
        logger.info(f"Position sizing: size={risk_adjusted_size:.4f}, "
                   f"kelly={adjusted_kelly:.4f}, edge={expected_edge:.4f}")
        
        return signal

    def _calculate_kelly_fraction(self, expected_edge: float, volatility: float) -> float:
        """
        Calculate Kelly fraction: f* = E/σ²
        
        Args:
            expected_edge: Expected return per unit risk
            volatility: Annualized volatility
            
        Returns:
            Optimal Kelly fraction
        """
        if volatility <= 0:
            return self.min_kelly_fraction
            
        # Kelly formula: f* = μ/σ² where μ is expected return, σ is volatility
        kelly_fraction = expected_edge / (volatility ** 2)
        
        # Apply fractional Kelly (typically 25-50% of full Kelly)
        fractional_kelly = kelly_fraction * 0.25
        
        return np.clip(fractional_kelly, self.min_kelly_fraction, self.max_kelly_fraction)

    def _calculate_current_volatility(self) -> float:
        """Calculate current annualized volatility from price history."""
        if len(self.price_history) < self.volatility_window:
            return 0.20  # Default 20% volatility
        
        prices = np.array(list(self.price_history)[-self.volatility_window:])
        returns = np.diff(np.log(prices))
        
        # Annualize volatility (assuming minute-level data)
        daily_vol = np.std(returns) * np.sqrt(24 * 60)  # Minutes to daily
        annual_vol = daily_vol * np.sqrt(365)
        
        return annual_vol

    def _get_performance_adjustment(self) -> float:
        """
        Get performance adjustment factor based on recent trading results.
        
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        if len(self.trade_history) < 10:
            return 1.0  # Neutral adjustment
        
        # Calculate recent performance metrics
        recent_expectancy = self._calculate_expectancy()
        recent_sharpe = self._calculate_sharpe_ratio()
        
        # Adjust based on performance
        expectancy_adjustment = 1.0 + (recent_expectancy * 0.5)  # Scale expectancy impact
        sharpe_adjustment = 1.0 + (recent_sharpe * 0.1)  # Scale Sharpe impact
        
        # Combine adjustments
        combined_adjustment = (expectancy_adjustment + sharpe_adjustment) / 2
        
        return np.clip(combined_adjustment, 0.5, 1.5)

    def _calculate_expectancy(self) -> float:
        """
        Calculate rolling expectancy: E[P/L] = P̄/N - ½σ²
        
        Returns:
            Current expectancy
        """
        if len(self.trade_history) < 5:
            return 0.0
        
        returns = np.array([trade['pnl'] for trade in self.trade_history])
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        # Expectancy formula with variance penalty
        expectancy = mean_return - 0.5 * variance
        
        return expectancy

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent returns."""
        if len(self.return_history) < 10:
            return 0.0
        
        returns = np.array(list(self.return_history))
        excess_returns = returns - (self.risk_free_rate / 365)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
        return sharpe

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if len(self.trade_history) < 5:
            return 0.5  # Default 50% win rate
        
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return wins / len(self.trade_history)

    def update_price(self, price: float):
        """Update price history for volatility calculation."""
        self.price_history.append(price)

    def record_trade(self, entry_price: float, exit_price: float, position_size: float,
                    trade_type: str, duration: float):
        """
        Record a completed trade for performance tracking.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            trade_type: 'long' or 'short'
            duration: Trade duration in seconds
        """
        # Calculate P&L
        if trade_type == 'long':
            pnl = (exit_price - entry_price) / entry_price * position_size
        else:
            pnl = (entry_price - exit_price) / entry_price * position_size
        
        # Record trade
        trade_record = {
            'timestamp': time.time(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'trade_type': trade_type,
            'duration': duration,
            'pnl': pnl,
            'return': pnl / position_size if position_size > 0 else 0
        }
        
        self.trade_history.append(trade_record)
        self.return_history.append(trade_record['return'])
        
        # Update equity and drawdown
        self.current_equity *= (1 + trade_record['return'])
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        logger.info(f"Trade recorded: P&L={pnl:.4f}, Return={trade_record['return']:.4f}, "
                   f"Drawdown={self.current_drawdown:.4f}")

    def should_reduce_risk(self) -> bool:
        """Check if risk should be reduced due to poor performance."""
        # Reduce risk if in significant drawdown
        if self.current_drawdown > self.max_drawdown_limit * 0.6:
            return True
        
        # Reduce risk if recent expectancy is negative
        if len(self.trade_history) >= 10 and self._calculate_expectancy() < -0.02:
            return True
        
        # Reduce risk if Sharpe ratio is very poor
        if len(self.return_history) >= 20 and self._calculate_sharpe_ratio() < -0.5:
            return True
        
        return False

    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return RiskMetrics(
            current_expectancy=self._calculate_expectancy(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            max_drawdown=self.current_drawdown,
            win_rate=self._calculate_win_rate(),
            profit_factor=self._calculate_profit_factor(),
            kelly_optimal=self._calculate_kelly_fraction(self._calculate_expectancy(), 
                                                        self._calculate_current_volatility()),
            current_volatility=self._calculate_current_volatility(),
            var_95=self._calculate_var_95()
        )

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.trade_history) < 5:
            return 1.0
        
        profits = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        losses = sum(abs(trade['pnl']) for trade in self.trade_history if trade['pnl'] < 0)
        
        if losses == 0:
            return float('inf') if profits > 0 else 1.0
        
        return profits / losses

    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk."""
        if len(self.return_history) < 20:
            return 0.05  # Default 5% VaR
        
        returns = np.array(list(self.return_history))
        return np.percentile(returns, 5)  # 5th percentile for 95% VaR

    def reset_performance(self):
        """Reset performance tracking (use with caution)."""
        self.current_drawdown = 0.0
        self.peak_equity = self.current_equity
        logger.info("Risk engine performance metrics reset") 