#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Manager 🛡️

Comprehensive risk assessment and management for Schwabot trading system:
• Real-time risk assessment and position sizing
• Portfolio risk monitoring and alerts
• GPU/CPU tensor operations for risk calculations
• Multi-dimensional risk metrics (VaR, CVaR, Sharpe ratio)
• Dynamic risk limits and circuit breakers

Features:
- GPU-accelerated risk calculations with automatic CPU fallback
- Real-time portfolio monitoring and risk alerts
- Advanced risk metrics (VaR, CVaR, maximum drawdown)
- Position sizing based on risk tolerance
- Circuit breakers and emergency stops
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import cupy as cp
    import numpy as np
    USING_CUDA = True
    xp = cp
    _backend = 'cupy (GPU)'
except ImportError:
    try:
        import numpy as np
        USING_CUDA = False
        xp = np
        _backend = 'numpy (CPU)'
    except ImportError:
        xp = None
        _backend = 'none'

logger = logging.getLogger(__name__)
if xp is None:
    logger.warning("❌ NumPy not available for risk calculations")
else:
    logger.info(f"⚡ RiskManager using {_backend} for tensor operations")


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingMode(Enum):
    """Processing mode for risk calculations."""
    GPU_ACCELERATED = "gpu_accelerated"
    CPU_FALLBACK = "cpu_fallback"
    HYBRID = "hybrid"
    SAFE_MODE = "safe_mode"


@dataclass
class RiskMetric:
    """Risk metric with tensor math integration."""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    cvar_95: float  # Conditional Value at Risk (95% confidence)
    cvar_99: float  # Conditional Value at Risk (99% confidence)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    skewness: float
    kurtosis: float
    timestamp: float = field(default_factory=time.time)
    tensor_confidence: float = 0.0


@dataclass
class PositionRisk:
    """Position-specific risk assessment."""
    symbol: str
    position_size: float
    current_value: float
    unrealized_pnl: float
    risk_metrics: RiskMetric
    risk_level: RiskLevel
    max_position_size: float
    stop_loss_level: float
    take_profit_level: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment."""
    total_value: float
    total_pnl: float
    risk_metrics: RiskMetric
    risk_level: RiskLevel
    positions: List[PositionRisk]
    correlation_matrix: xp.ndarray
    covariance_matrix: xp.ndarray
    timestamp: float = field(default_factory=time.time)


class RiskManager:
    """
    Comprehensive risk management system with tensor math integration.
    Handles real-time risk assessment, position sizing, and portfolio monitoring.
    """
    def __init__(self, risk_tolerance: float = 0.02, max_portfolio_risk: float = 0.05):
        self.risk_tolerance = risk_tolerance  # 2% default
        self.max_portfolio_risk = max_portfolio_risk  # 5% default
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_history: List[PortfolioRisk] = []
        self.circuit_breakers: Dict[str, bool] = {}
        self.processing_mode = ProcessingMode.GPU_ACCELERATED if USING_CUDA else ProcessingMode.CPU_FALLBACK
        self.alert_thresholds = {
            RiskLevel.LOW: 0.01,
            RiskLevel.MEDIUM: 0.03,
            RiskLevel.HIGH: 0.05,
            RiskLevel.CRITICAL: 0.10
        }

    def calculate_risk_metrics(self, returns: xp.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> RiskMetric:
        """Calculate comprehensive risk metrics using tensor operations."""
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")
            
            # Basic statistics
            mean_return = xp.mean(returns)
            volatility = xp.std(returns)
            
            # Value at Risk (VaR)
            var_95 = float(xp.percentile(returns, 5))  # 95% VaR
            var_99 = float(xp.percentile(returns, 1))  # 99% VaR
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = float(xp.mean(returns[returns <= var_95]))
            cvar_99 = float(xp.mean(returns[returns <= var_99]))
            
            # Sharpe and Sortino ratios
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < mean_return]
            downside_deviation = xp.std(downside_returns) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = xp.cumprod(1 + returns)
            running_max = xp.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = float(xp.min(drawdowns))
            
            # Higher moments
            skewness = float(xp.mean(((returns - mean_return) / volatility) ** 3)) if volatility > 0 else 0
            kurtosis = float(xp.mean(((returns - mean_return) / volatility) ** 4)) if volatility > 0 else 0
            
            # Beta and correlation (simplified - would need market data)
            beta = 1.0  # Default beta
            correlation = 0.0  # Default correlation
            
            # Tensor confidence based on data quality
            tensor_confidence = min(1.0, len(returns) / 1000.0)  # More data = higher confidence
            
            return RiskMetric(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                sharpe_ratio=float(sharpe_ratio),
                sortino_ratio=float(sortino_ratio),
                max_drawdown=max_drawdown,
                volatility=float(volatility),
                beta=beta,
                correlation=correlation,
                skewness=skewness,
                kurtosis=kurtosis,
                tensor_confidence=tensor_confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return RiskMetric(
                var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                volatility=0.0, beta=1.0, correlation=0.0,
                skewness=0.0, kurtosis=0.0, tensor_confidence=0.0
            )

    def assess_position_risk(self, symbol: str, position_size: float, current_price: float, 
                           historical_returns: xp.ndarray, entry_price: float) -> PositionRisk:
        """Assess risk for a specific position."""
        try:
            # Calculate position metrics
            current_value = position_size * current_price
            unrealized_pnl = position_size * (current_price - entry_price)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(historical_returns)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            # Calculate position limits
            max_position_size = self._calculate_max_position_size(risk_metrics, current_value)
            
            # Calculate stop loss and take profit levels
            stop_loss_level = entry_price * (1 - self.risk_tolerance)
            take_profit_level = entry_price * (1 + self.risk_tolerance * 2)  # 2:1 reward/risk
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                max_position_size=max_position_size,
                stop_loss_level=stop_loss_level,
                take_profit_level=take_profit_level
            )
            
            self.positions[symbol] = position_risk
            return position_risk
            
        except Exception as e:
            logger.error(f"❌ Failed to assess position risk for {symbol}: {e}")
            return None

    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRisk:
        """Assess overall portfolio risk."""
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")
            
            total_value = portfolio_data.get('total_value', 0.0)
            total_pnl = portfolio_data.get('total_pnl', 0.0)
            positions_data = portfolio_data.get('positions', [])
            
            # Calculate portfolio returns
            portfolio_returns = xp.array(portfolio_data.get('returns', []))
            if len(portfolio_returns) == 0:
                portfolio_returns = xp.zeros(100)  # Default empty portfolio
            
            # Calculate portfolio risk metrics
            risk_metrics = self.calculate_risk_metrics(portfolio_returns)
            
            # Determine portfolio risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            # Calculate correlation and covariance matrices
            symbols = [pos['symbol'] for pos in positions_data]
            if len(symbols) > 1:
                returns_matrix = xp.array([pos.get('returns', []) for pos in positions_data])
                correlation_matrix = xp.corrcoef(returns_matrix) if returns_matrix.shape[0] > 1 else xp.eye(len(symbols))
                covariance_matrix = xp.cov(returns_matrix) if returns_matrix.shape[0] > 1 else xp.eye(len(symbols))
            else:
                correlation_matrix = xp.eye(len(symbols))
                covariance_matrix = xp.eye(len(symbols))
            
            # Create position risk objects
            positions = []
            for pos_data in positions_data:
                symbol = pos_data['symbol']
                if symbol in self.positions:
                    positions.append(self.positions[symbol])
            
            portfolio_risk = PortfolioRisk(
                total_value=total_value,
                total_pnl=total_pnl,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                positions=positions,
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix
            )
            
            self.risk_history.append(portfolio_risk)
            
            # Check for circuit breakers
            self._check_circuit_breakers(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"❌ Failed to assess portfolio risk: {e}")
            return None

    def _determine_risk_level(self, risk_metrics: RiskMetric) -> RiskLevel:
        """Determine risk level based on metrics."""
        # Simple risk level determination based on VaR
        var_ratio = abs(risk_metrics.var_95)
        
        if var_ratio <= self.alert_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif var_ratio <= self.alert_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif var_ratio <= self.alert_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _calculate_max_position_size(self, risk_metrics: RiskMetric, current_value: float) -> float:
        """Calculate maximum position size based on risk metrics."""
        # Kelly Criterion inspired position sizing
        win_rate = 0.5  # Default win rate
        avg_win = abs(risk_metrics.var_95) * 2  # Assume 2:1 reward/risk
        avg_loss = abs(risk_metrics.var_95)
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        return current_value * kelly_fraction

    def _check_circuit_breakers(self, portfolio_risk: PortfolioRisk) -> None:
        """Check and trigger circuit breakers if needed."""
        if portfolio_risk.risk_level == RiskLevel.CRITICAL:
            self.circuit_breakers['portfolio'] = True
            logger.critical("🚨 CRITICAL RISK: Portfolio circuit breaker triggered!")
            
        if portfolio_risk.risk_metrics.max_drawdown < -0.20:  # 20% drawdown
            self.circuit_breakers['drawdown'] = True
            logger.critical("🚨 DRAWDOWN ALERT: 20% drawdown circuit breaker triggered!")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            if not self.risk_history:
                return {"error": "No risk history available"}
            
            latest_risk = self.risk_history[-1]
            
            return {
                "total_positions": len(latest_risk.positions),
                "total_value": latest_risk.total_value,
                "total_pnl": latest_risk.total_pnl,
                "risk_level": latest_risk.risk_level.value,
                "var_95": latest_risk.risk_metrics.var_95,
                "var_99": latest_risk.risk_metrics.var_99,
                "max_drawdown": latest_risk.risk_metrics.max_drawdown,
                "sharpe_ratio": latest_risk.risk_metrics.sharpe_ratio,
                "volatility": latest_risk.risk_metrics.volatility,
                "circuit_breakers_active": any(self.circuit_breakers.values()),
                "tensor_confidence": latest_risk.risk_metrics.tensor_confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get risk summary: {e}")
            return {"error": str(e)}

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self.circuit_breakers.clear()
        logger.info("✅ Circuit breakers reset")

    def set_risk_tolerance(self, tolerance: float) -> None:
        """Set risk tolerance level."""
        self.risk_tolerance = max(0.001, min(tolerance, 0.1))  # Between 0.1% and 10%
        logger.info(f"🔧 Risk tolerance set to {self.risk_tolerance:.3f}")


# Singleton instance for global use
risk_manager = RiskManager()