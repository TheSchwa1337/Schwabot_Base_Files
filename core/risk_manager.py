"""
Core Risk Management System for Quantitative Trading
Provides comprehensive risk management capabilities including position sizing,
dynamic stop-loss, portfolio risk controls, and real-time monitoring.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging
from enum import Enum

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class PositionRisk:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    risk_level: RiskLevel
    kelly_fraction: float
    max_drawdown: float
    volatility: float
    timestamp: datetime

@dataclass
class PortfolioRisk:
    total_exposure: float
    max_position_size: float
    current_drawdown: float
    portfolio_volatility: float
    risk_level: RiskLevel
    correlation_matrix: np.ndarray
    var_95: float  # Value at Risk at 95% confidence
    timestamp: datetime

class RiskManager:
    """
    Core risk management system for quantitative trading.
    Handles position sizing, stop-loss management, portfolio risk controls,
    and real-time risk monitoring.
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 max_position_size: float = 1.0,    # 100% max position size
                 max_drawdown: float = 0.1,         # 10% max drawdown
                 var_confidence: float = 0.95,      # 95% VaR confidence
                 update_interval: float = 1.0):     # 1 second update interval
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.update_interval = update_interval
        
        # Internal state
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None
        self.risk_history: List[PortfolioRisk] = []
        self.last_update = datetime.now()
        self.ghost_shell_stops = {}
        self.risk_metrics = {}
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion fraction for position sizing.
        
        Args:
            win_rate: Probability of winning trades
            win_loss_ratio: Ratio of winning trade size to losing trade size
            
        Returns:
            float: Kelly fraction (0 to 1)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return max(0.0, min(kelly, 0.5))  # Cap at 50% for safety
        
    def calculate_position_size(self,
                              symbol: str,
                              price: float,
                              volatility: float,
                              win_rate: float,
                              win_loss_ratio: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk constraints.
        """
        # Calculate base Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_rate, win_loss_ratio)
        
        # Adjust for volatility
        vol_adjustment = 1.0 / (1.0 + volatility)
        
        # Calculate final position size
        position_size = kelly_fraction * vol_adjustment * self.max_position_size
        
        # Ensure position size doesn't exceed portfolio limits
        if self.portfolio_risk:
            remaining_capacity = 1.0 - self.portfolio_risk.total_exposure
            position_size = min(position_size, remaining_capacity)
            
        # Adjust for active ghost shell stops
        if symbol in self.ghost_shell_stops:
            stop_info = self.ghost_shell_stops[symbol]
            stop_distance = abs(price - stop_info['stop_price']) / price
            
            # Reduce position size if stop is too close
            if stop_distance < 0.01:  # 1% minimum distance
                position_size *= 0.5
                
        return position_size
        
    def calculate_dynamic_stop_loss(self,
                                  symbol: str,
                                  price: float,
                                  volatility: float,
                                  atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels.
        """
        # Base stop-loss on ATR
        stop_distance = atr * 2.0  # 2 ATR units
        
        # Adjust for volatility
        vol_adjustment = 1.0 + volatility
        
        # Calculate stop-loss and take-profit
        stop_loss = price - (stop_distance * vol_adjustment)
        take_profit = price + (stop_distance * vol_adjustment * 1.5)  # 1.5x risk:reward
        
        # Integrate with ghost shell stops
        if symbol in self.ghost_shell_stops:
            ghost_stop = self.ghost_shell_stops[symbol]['stop_price']
            # Use tighter of the two stops
            stop_loss = max(stop_loss, ghost_stop)
            
        return stop_loss, take_profit
        
    def update_portfolio_risk(self, positions: Dict[str, PositionRisk]) -> PortfolioRisk:
        """
        Update portfolio-level risk metrics.
        """
        # Calculate total exposure
        total_exposure = sum(pos.size for pos in positions.values())
        
        # Calculate portfolio volatility
        returns = []
        for pos in positions.values():
            if pos.entry_price > 0:
                returns.append((pos.current_price - pos.entry_price) / pos.entry_price)
        
        portfolio_volatility = np.std(returns) if returns else 0.0
        
        # Calculate Value at Risk
        var_95 = self.calculate_var(returns) if returns else 0.0
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(positions)
        
        # Determine risk level
        risk_level = self.determine_risk_level(
            total_exposure,
            portfolio_volatility,
            var_95
        )
        
        # Create portfolio risk snapshot
        portfolio_risk = PortfolioRisk(
            total_exposure=total_exposure,
            max_position_size=self.max_position_size,
            current_drawdown=self.calculate_drawdown(positions),
            portfolio_volatility=portfolio_volatility,
            risk_level=risk_level,
            correlation_matrix=correlation_matrix,
            var_95=var_95,
            timestamp=datetime.now()
        )
        
        # Update state
        self.portfolio_risk = portfolio_risk
        self.risk_history.append(portfolio_risk)
        
        return portfolio_risk
        
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk at specified confidence level.
        """
        if not returns:
            return 0.0
            
        return np.percentile(returns, (1 - confidence) * 100)
        
    def calculate_correlation_matrix(self, positions: Dict[str, PositionRisk]) -> np.ndarray:
        """
        Calculate correlation matrix between positions.
        """
        # Implementation depends on available price data
        # This is a placeholder that returns identity matrix
        n = len(positions)
        return np.eye(n)
        
    def calculate_drawdown(self, positions: Dict[str, PositionRisk]) -> float:
        """
        Calculate current portfolio drawdown.
        """
        if not positions:
            return 0.0
            
        total_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.size
            for pos in positions.values()
        )
        
        peak_value = max(
            sum(pos.entry_price * pos.size for pos in positions.values()),
            total_pnl
        )
        
        return (peak_value - total_pnl) / peak_value if peak_value > 0 else 0.0
        
    def determine_risk_level(self,
                           total_exposure: float,
                           volatility: float,
                           var_95: float) -> RiskLevel:
        """
        Determine current portfolio risk level.
        """
        if total_exposure > 0.9 or volatility > 0.3 or var_95 > 0.1:
            return RiskLevel.EXTREME
        elif total_exposure > 0.7 or volatility > 0.2 or var_95 > 0.05:
            return RiskLevel.HIGH
        elif total_exposure > 0.5 or volatility > 0.1 or var_95 > 0.02:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def check_risk_limits(self, symbol: str, new_position: PositionRisk) -> bool:
        """
        Check if new position would violate risk limits.
        """
        if not self.portfolio_risk:
            return True
            
        # Check position size limit
        if new_position.size > self.max_position_size:
            self.logger.warning(f"Position size {new_position.size} exceeds limit {self.max_position_size}")
            return False
            
        # Check portfolio exposure
        new_exposure = self.portfolio_risk.total_exposure + new_position.size
        if new_exposure > 1.0:
            self.logger.warning(f"New exposure {new_exposure} would exceed 100%")
            return False
            
        # Check drawdown limit
        if self.portfolio_risk.current_drawdown > self.max_drawdown:
            self.logger.warning(f"Current drawdown {self.portfolio_risk.current_drawdown} exceeds limit {self.max_drawdown}")
            return False
            
        return True
        
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.
        """
        if not self.portfolio_risk:
            return {"error": "No portfolio risk data available"}
            
        return {
            "timestamp": self.portfolio_risk.timestamp.isoformat(),
            "total_exposure": self.portfolio_risk.total_exposure,
            "current_drawdown": self.portfolio_risk.current_drawdown,
            "portfolio_volatility": self.portfolio_risk.portfolio_volatility,
            "risk_level": self.portfolio_risk.risk_level.name,
            "var_95": self.portfolio_risk.var_95,
            "position_risks": {
                symbol: {
                    "size": pos.size,
                    "risk_level": pos.risk_level.name,
                    "kelly_fraction": pos.kelly_fraction,
                    "max_drawdown": pos.max_drawdown,
                    "volatility": pos.volatility
                }
                for symbol, pos in self.positions.items()
            }
        }
        
    def integrate_ghost_shell_stop(self, symbol: str, stop_info: Dict):
        """Integrate ghost shell stop loss into risk management"""
        self.ghost_shell_stops[symbol] = stop_info
        
        # Calculate potential loss
        current_price = self.get_current_price(symbol)
        stop_price = stop_info['stop_price']
        quantity = stop_info['quantity']
        
        potential_loss = abs(current_price - stop_price) * quantity
        
        # Update risk metrics
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = {}
            
        self.risk_metrics[symbol].update({
            'potential_loss': potential_loss,
            'stop_distance': abs(current_price - stop_price) / current_price,
            'last_update': datetime.now()
        })
        
        # Check if stop needs adjustment
        self._check_stop_adjustment(symbol)
        
    def _check_stop_adjustment(self, symbol: str):
        """Check if ghost shell stop needs adjustment based on risk metrics"""
        if symbol not in self.risk_metrics:
            return
            
        metrics = self.risk_metrics[symbol]
        stop_info = self.ghost_shell_stops[symbol]
        
        # Calculate volatility
        volatility = self.calculate_volatility(symbol)
        
        # Adjust stop if volatility has changed significantly
        if volatility > metrics.get('last_volatility', 0) * 1.5:
            new_stop = self._calculate_adjusted_stop(symbol, volatility)
            if new_stop != stop_info['stop_price']:
                self._update_stop_price(symbol, new_stop)
                
        # Update last volatility
        metrics['last_volatility'] = volatility
        
    def _calculate_adjusted_stop(self, symbol: str, volatility: float) -> float:
        """Calculate adjusted stop price based on volatility"""
        current_price = self.get_current_price(symbol)
        base_distance = volatility * 2.0  # 2x volatility as base distance
        
        # Adjust for market conditions
        if self.is_market_volatile():
            base_distance *= 1.5
            
        return current_price - base_distance
        
    def _update_stop_price(self, symbol: str, new_stop: float):
        """Update ghost shell stop price"""
        try:
            # Cancel existing stop
            self.cancel_ghost_shell_stop(symbol)
            
            # Place new stop
            stop_info = self.ghost_shell_stops[symbol]
            self.place_ghost_shell_stop(
                symbol=symbol,
                stop_price=new_stop,
                quantity=stop_info['quantity']
            )
            
            self.logger.info(f"Updated ghost shell stop for {symbol} to {new_stop}")
            
        except Exception as e:
            self.logger.error(f"Error updating stop price: {str(e)}")
            
    def is_market_volatile(self) -> bool:
        """Check if market is currently volatile"""
        # Implement market volatility check logic
        return False  # Placeholder
        
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate current market volatility"""
        # Implement volatility calculation logic
        return 0.0  # Placeholder
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        # Implement price fetching logic
        return 0.0  # Placeholder 