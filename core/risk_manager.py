#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Manager 🛡️ - GODMODE ENHANCED

Comprehensive risk assessment and management for Schwabot trading system:
• Real-time risk assessment and position sizing
• Portfolio risk monitoring and alerts
• GPU/CPU tensor operations for risk calculations
• Multi-dimensional risk metrics (VaR, CVaR, Sharpe ratio, MDD)
• Dynamic risk limits and circuit breakers
• Schwabot strategy integration with hash outputs

Mathematical Foundation:
- Value at Risk (VaR): VaR = μ - z_α * σ
- Expected Shortfall (ES): ES = E[X|X > VaR]
- Sharpe Ratio: Sharpe = (R_p - R_f) / σ_p
- Maximum Drawdown: MDD = max((Peak - Trough) / Peak)
- Kelly Criterion: f* = (bp - q) / b

Features:
- GPU-accelerated risk calculations with automatic CPU fallback
- Real-time portfolio monitoring and risk alerts
- Advanced risk metrics (VaR, CVaR, maximum drawdown)
- Position sizing based on risk tolerance
- Circuit breakers and emergency stops
- JSON-compatible risk flags for strategy integration
- Hash-based risk state tracking for Schwabot decision logic
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Risk level enumeration for Schwabot decision logic."""
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
    """
    Risk metric with tensor math integration and Schwabot compatibility.
    
    Mathematical Formulas:
    - VaR: VaR = μ - z_α * σ (where z_α is the α-quantile of standard normal)
    - CVaR: CVaR = E[X|X > VaR] (Expected Shortfall)
    - Sharpe: Sharpe = (R_p - R_f) / σ_p (risk-adjusted return)
    - MDD: MDD = max((Peak - Trough) / Peak) (maximum drawdown)
    """
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    cvar_95: float  # Conditional Value at Risk (95% confidence) = Expected Shortfall
    cvar_99: float  # Conditional Value at Risk (99% confidence) = Expected Shortfall
    sharpe_ratio: float  # Sharpe ratio (risk-adjusted return)
    sortino_ratio: float  # Sortino ratio (downside risk-adjusted return)
    max_drawdown: float  # Maximum drawdown
    volatility: float  # Standard deviation of returns
    beta: float  # Beta coefficient (market sensitivity)
    correlation: float  # Correlation with market
    skewness: float  # Third moment (distribution asymmetry)
    kurtosis: float  # Fourth moment (distribution tails)
    timestamp: float = field(default_factory=time.time)
    tensor_confidence: float = 0.0  # Confidence in tensor calculations
    risk_hash: str = ""  # Hash for Schwabot strategy integration


@dataclass
class PositionRisk:
    """Position-specific risk assessment with Schwabot integration."""
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
    position_hash: str = ""  # Hash for Schwabot decision logic


@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment with Schwabot integration."""
    total_value: float
    total_pnl: float
    risk_metrics: RiskMetric
    risk_level: RiskLevel
    positions: List[PositionRisk]
    correlation_matrix: xp.ndarray
    covariance_matrix: xp.ndarray
    timestamp: float = field(default_factory=time.time)
    portfolio_hash: str = ""  # Hash for Schwabot strategy integration


class RiskManager:
    """
    Comprehensive risk management system with tensor math integration and Schwabot strategy compatibility.
    
    Handles real-time risk assessment, position sizing, and portfolio monitoring.
    Integrates with Schwabot's decision logic through hash-based state tracking.
    
    Mathematical Foundation:
    - VaR: VaR = μ - z_α * σ
    - Expected Shortfall: ES = E[X|X > VaR]
    - Sharpe Ratio: Sharpe = (R_p - R_f) / σ_p
    - Maximum Drawdown: MDD = max((Peak - Trough) / Peak)
    - Kelly Criterion: f* = (bp - q) / b
    """
    
    def __init__(self, risk_tolerance: float = 0.02, max_portfolio_risk: float = 0.05):
        """
        Initialize RiskManager with Schwabot-compatible configuration.
        
        Args:
            risk_tolerance: Maximum acceptable risk per position (default: 2%)
            max_portfolio_risk: Maximum acceptable portfolio risk (default: 5%)
        """
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
        
        logger.info(f"🛡️ RiskManager initialized with {_backend} backend")
        logger.info(f"   Risk tolerance: {risk_tolerance:.1%}")
        logger.info(f"   Max portfolio risk: {max_portfolio_risk:.1%}")

    def compute_var(self, returns: xp.ndarray, confidence_level: float = 0.95) -> float:
        """
        Compute Value at Risk (VaR) using historical simulation.
        
        Mathematical Formula: VaR = μ - z_α * σ
        Where:
        - μ = mean return
        - z_α = α-quantile of standard normal distribution
        - σ = standard deviation of returns
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value (negative for losses)
            
        Raises:
            ValueError: If confidence_level is not in (0, 1) or returns is empty
        """
        try:
            if not (0 < confidence_level < 1):
                raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
            
            if len(returns) == 0:
                raise ValueError("Returns array cannot be empty")
            
            # Historical simulation approach (more robust than parametric)
            percentile = (1 - confidence_level) * 100
            var = float(xp.percentile(returns, percentile))
            
            logger.debug(f"VaR({confidence_level:.0%}) = {var:.4f}")
            return var
            
        except Exception as e:
            logger.error(f"❌ Failed to compute VaR: {e}")
            raise

    def compute_expected_shortfall(self, returns: xp.ndarray, confidence_level: float = 0.95) -> float:
        """
        Compute Expected Shortfall (ES) / Conditional Value at Risk (CVaR).
        
        Mathematical Formula: ES = E[X|X > VaR]
        Where:
        - X = return distribution
        - VaR = Value at Risk at given confidence level
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95% ES)
            
        Returns:
            Expected Shortfall value (negative for losses)
            
        Raises:
            ValueError: If confidence_level is not in (0, 1) or returns is empty
        """
        try:
            if not (0 < confidence_level < 1):
                raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
            
            if len(returns) == 0:
                raise ValueError("Returns array cannot be empty")
            
            # Compute VaR first
            var = self.compute_var(returns, confidence_level)
            
            # Compute ES as mean of returns beyond VaR
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                # If no returns beyond VaR, ES = VaR
                es = var
            else:
                es = float(xp.mean(tail_returns))
            
            logger.debug(f"ES({confidence_level:.0%}) = {es:.4f}")
            return es
            
        except Exception as e:
            logger.error(f"❌ Failed to compute Expected Shortfall: {e}")
            raise

    def compute_sharpe_ratio(self, returns: xp.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Compute Sharpe ratio (risk-adjusted return).
        
        Mathematical Formula: Sharpe = (R_p - R_f) / σ_p
        Where:
        - R_p = portfolio return
        - R_f = risk-free rate
        - σ_p = portfolio standard deviation
        
        Args:
            returns: Array of historical returns
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio (higher is better)
            
        Raises:
            ValueError: If returns array is empty
        """
        try:
            if len(returns) == 0:
                raise ValueError("Returns array cannot be empty")
            
            # Annualize returns and risk-free rate (assuming daily data)
            annualized_return = float(xp.mean(returns) * 252)  # 252 trading days
            annualized_volatility = float(xp.std(returns) * xp.sqrt(252))
            
            # Compute Sharpe ratio
            if annualized_volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            else:
                sharpe_ratio = 0.0
            
            logger.debug(f"Sharpe ratio = {sharpe_ratio:.4f}")
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"❌ Failed to compute Sharpe ratio: {e}")
            raise

    def compute_max_drawdown(self, returns: xp.ndarray) -> float:
        """
        Compute Maximum Drawdown (MDD).
        
        Mathematical Formula: MDD = max((Peak - Trough) / Peak)
        Where:
        - Peak = running maximum of cumulative returns
        - Trough = minimum value after each peak
        
        Args:
            returns: Array of historical returns
            
        Returns:
            Maximum drawdown (negative value, e.g., -0.15 for 15% drawdown)
            
        Raises:
            ValueError: If returns array is empty
        """
        try:
            if len(returns) == 0:
                raise ValueError("Returns array cannot be empty")
            
            # Compute cumulative returns
            cumulative_returns = xp.cumprod(1 + returns)
            
            # Compute running maximum
            running_max = xp.maximum.accumulate(cumulative_returns)
            
            # Compute drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = float(xp.min(drawdowns))
            
            logger.debug(f"Maximum drawdown = {max_drawdown:.4f}")
            return max_drawdown
            
        except Exception as e:
            logger.error(f"❌ Failed to compute maximum drawdown: {e}")
            raise

    def calculate_risk_metrics(self, returns: xp.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> RiskMetric:
        """
        Calculate comprehensive risk metrics using tensor operations.
        
        Implements all core risk metrics with proper mathematical formulas:
        - VaR: VaR = μ - z_α * σ
        - Expected Shortfall: ES = E[X|X > VaR]
        - Sharpe Ratio: Sharpe = (R_p - R_f) / σ_p
        - Maximum Drawdown: MDD = max((Peak - Trough) / Peak)
        
        Args:
            returns: Array of historical returns
            confidence_levels: List of confidence levels for VaR/ES calculations
            
        Returns:
            RiskMetric object with all computed metrics
            
        Raises:
            ValueError: If tensor operations not available or invalid inputs
        """
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")
            
            if len(returns) == 0:
                raise ValueError("Returns array cannot be empty")
            
            # Basic statistics
            mean_return = float(xp.mean(returns))
            volatility = float(xp.std(returns))
            
            # Value at Risk (VaR) - using individual function
            var_95 = self.compute_var(returns, 0.95)
            var_99 = self.compute_var(returns, 0.99)
            
            # Expected Shortfall (ES) - using individual function
            cvar_95 = self.compute_expected_shortfall(returns, 0.95)
            cvar_99 = self.compute_expected_shortfall(returns, 0.99)
            
            # Sharpe and Sortino ratios
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = self.compute_sharpe_ratio(returns, risk_free_rate)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < mean_return]
            if len(downside_returns) > 0:
                downside_deviation = float(xp.std(downside_returns) * xp.sqrt(252))
                sortino_ratio = (mean_return * 252 - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = 0.0
            
            # Maximum drawdown - using individual function
            max_drawdown = self.compute_max_drawdown(returns)
            
            # Higher moments
            if volatility > 0:
                standardized_returns = (returns - mean_return) / volatility
                skewness = float(xp.mean(standardized_returns ** 3))
                kurtosis = float(xp.mean(standardized_returns ** 4))
            else:
                skewness = 0.0
                kurtosis = 0.0
            
            # Beta and correlation (simplified - would need market data)
            beta = 1.0  # Default beta
            correlation = 0.0  # Default correlation
            
            # Tensor confidence based on data quality
            tensor_confidence = min(1.0, len(returns) / 1000.0)  # More data = higher confidence
            
            # Generate risk hash for Schwabot integration
            risk_data = {
                "var_95": var_95, "var_99": var_99,
                "cvar_95": cvar_95, "cvar_99": cvar_99,
                "sharpe": sharpe_ratio, "max_dd": max_drawdown,
                "vol": volatility, "timestamp": time.time()
            }
            risk_hash = hashlib.sha256(json.dumps(risk_data, sort_keys=True).encode()).hexdigest()[:16]
            
            risk_metric = RiskMetric(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                skewness=skewness,
                kurtosis=kurtosis,
                tensor_confidence=tensor_confidence,
                risk_hash=risk_hash
            )
            
            logger.info(f"✅ Risk metrics calculated - VaR95: {var_95:.4f}, Sharpe: {sharpe_ratio:.4f}, MDD: {max_drawdown:.4f}")
            return risk_metric
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return RiskMetric(
                var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                volatility=0.0, beta=1.0, correlation=0.0,
                skewness=0.0, kurtosis=0.0, tensor_confidence=0.0,
                risk_hash=""
            )

    def assess_position_risk(self, symbol: str, position_size: float, current_price: float, 
                           historical_returns: xp.ndarray, entry_price: float) -> PositionRisk:
        """
        Assess risk for a specific position with Schwabot integration.
        
        Args:
            symbol: Trading symbol
            position_size: Current position size
            current_price: Current market price
            historical_returns: Historical return data
            entry_price: Position entry price
            
        Returns:
            PositionRisk object with comprehensive risk assessment
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if position_size <= 0:
                raise ValueError(f"Position size must be positive, got {position_size}")
            if current_price <= 0:
                raise ValueError(f"Current price must be positive, got {current_price}")
            if entry_price <= 0:
                raise ValueError(f"Entry price must be positive, got {entry_price}")
            
            # Calculate position metrics
            current_value = position_size * current_price
            unrealized_pnl = position_size * (current_price - entry_price)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(historical_returns)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            # Calculate position limits using Kelly Criterion
            max_position_size = self._calculate_max_position_size(risk_metrics, current_value)
            
            # Calculate dynamic stop loss and take profit levels
            stop_loss_level = entry_price * (1 - self.risk_tolerance)
            take_profit_level = entry_price * (1 + self.risk_tolerance * 2)  # 2:1 reward/risk
            
            # Generate position hash for Schwabot integration
            position_data = {
                "symbol": symbol,
                "size": position_size,
                "value": current_value,
                "pnl": unrealized_pnl,
                "risk_level": risk_level.value,
                "var_95": risk_metrics.var_95,
                "sharpe": risk_metrics.sharpe_ratio,
                "timestamp": time.time()
            }
            position_hash = hashlib.sha256(json.dumps(position_data, sort_keys=True).encode()).hexdigest()[:16]
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                max_position_size=max_position_size,
                stop_loss_level=stop_loss_level,
                take_profit_level=take_profit_level,
                position_hash=position_hash
            )
            
            self.positions[symbol] = position_risk
            logger.info(f"✅ Position risk assessed for {symbol} - Risk: {risk_level.value}, PnL: {unrealized_pnl:.2f}")
            return position_risk
            
        except Exception as e:
            logger.error(f"❌ Failed to assess position risk for {symbol}: {e}")
            return None

    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRisk:
        """
        Assess overall portfolio risk with Schwabot integration.
        
        Args:
            portfolio_data: Dictionary containing portfolio information
                - total_value: Total portfolio value
                - total_pnl: Total portfolio PnL
                - positions: List of position dictionaries
                - returns: Portfolio return history
                
        Returns:
            PortfolioRisk object with comprehensive portfolio assessment
            
        Raises:
            ValueError: If portfolio data is invalid
        """
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
            
            # Generate portfolio hash for Schwabot integration
            portfolio_data_hash = {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "risk_level": risk_level.value,
                "var_95": risk_metrics.var_95,
                "sharpe": risk_metrics.sharpe_ratio,
                "max_dd": risk_metrics.max_drawdown,
                "positions_count": len(positions),
                "timestamp": time.time()
            }
            portfolio_hash = hashlib.sha256(json.dumps(portfolio_data_hash, sort_keys=True).encode()).hexdigest()[:16]
            
            portfolio_risk = PortfolioRisk(
                total_value=total_value,
                total_pnl=total_pnl,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                positions=positions,
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix,
                portfolio_hash=portfolio_hash
            )
            
            self.risk_history.append(portfolio_risk)
            
            # Check for circuit breakers
            self._check_circuit_breakers(portfolio_risk)
            
            logger.info(f"✅ Portfolio risk assessed - Value: {total_value:.2f}, Risk: {risk_level.value}, PnL: {total_pnl:.2f}")
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

    def get_risk_flags_json(self, portfolio_risk: Optional[PortfolioRisk] = None) -> Dict[str, Any]:
        """
        Get risk flags in JSON format for Schwabot strategy integration.
        
        Args:
            portfolio_risk: Optional portfolio risk object (uses latest if None)
            
        Returns:
            Dictionary with risk flags for strategy decision making
        """
        try:
            if portfolio_risk is None:
                if not self.risk_history:
                    return {"error": "No risk history available"}
                portfolio_risk = self.risk_history[-1]
            
            risk_flags = {
                "risk_level": portfolio_risk.risk_level.value,
                "var_95": portfolio_risk.risk_metrics.var_95,
                "var_99": portfolio_risk.risk_metrics.var_99,
                "expected_shortfall_95": portfolio_risk.risk_metrics.cvar_95,
                "expected_shortfall_99": portfolio_risk.risk_metrics.cvar_99,
                "sharpe_ratio": portfolio_risk.risk_metrics.sharpe_ratio,
                "sortino_ratio": portfolio_risk.risk_metrics.sortino_ratio,
                "max_drawdown": portfolio_risk.risk_metrics.max_drawdown,
                "volatility": portfolio_risk.risk_metrics.volatility,
                "total_value": portfolio_risk.total_value,
                "total_pnl": portfolio_risk.total_pnl,
                "positions_count": len(portfolio_risk.positions),
                "circuit_breakers_active": any(self.circuit_breakers.values()),
                "tensor_confidence": portfolio_risk.risk_metrics.tensor_confidence,
                "risk_hash": portfolio_risk.risk_metrics.risk_hash,
                "portfolio_hash": portfolio_risk.portfolio_hash,
                "timestamp": portfolio_risk.timestamp
            }
            
            return risk_flags
            
        except Exception as e:
            logger.error(f"❌ Failed to get risk flags: {e}")
            return {"error": str(e)}

    def get_strategy_decision_packet(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy decision packet for Schwabot integration.
        
        Returns:
            Dictionary with all risk information needed for strategy decisions
        """
        try:
            if not self.risk_history:
                return {"error": "No risk history available"}
            
            latest_risk = self.risk_history[-1]
            risk_flags = self.get_risk_flags_json(latest_risk)
            
            # Add strategy-specific flags
            strategy_packet = {
                **risk_flags,
                "can_trade": self._can_trade(latest_risk),
                "position_size_multiplier": self._get_position_size_multiplier(latest_risk),
                "stop_loss_multiplier": self._get_stop_loss_multiplier(latest_risk),
                "take_profit_multiplier": self._get_take_profit_multiplier(latest_risk),
                "risk_allocations": self._get_risk_allocations(latest_risk),
                "emergency_flags": self._get_emergency_flags(latest_risk)
            }
            
            return strategy_packet
            
        except Exception as e:
            logger.error(f"❌ Failed to get strategy decision packet: {e}")
            return {"error": str(e)}

    def _can_trade(self, portfolio_risk: PortfolioRisk) -> bool:
        """Determine if trading is allowed based on current risk."""
        return (
            portfolio_risk.risk_level != RiskLevel.CRITICAL and
            not any(self.circuit_breakers.values()) and
            portfolio_risk.risk_metrics.max_drawdown > -0.20  # 20% drawdown limit
        )

    def _get_position_size_multiplier(self, portfolio_risk: PortfolioRisk) -> float:
        """Get position size multiplier based on risk level."""
        multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.0
        }
        return multipliers.get(portfolio_risk.risk_level, 0.0)

    def _get_stop_loss_multiplier(self, portfolio_risk: PortfolioRisk) -> float:
        """Get stop loss multiplier based on volatility."""
        base_multiplier = 1.0
        volatility_factor = portfolio_risk.risk_metrics.volatility * 10  # Scale volatility
        return base_multiplier * (1 + volatility_factor)

    def _get_take_profit_multiplier(self, portfolio_risk: PortfolioRisk) -> float:
        """Get take profit multiplier based on Sharpe ratio."""
        base_multiplier = 2.0  # 2:1 reward/risk default
        sharpe_factor = max(0, portfolio_risk.risk_metrics.sharpe_ratio) * 0.5
        return base_multiplier + sharpe_factor

    def _get_risk_allocations(self, portfolio_risk: PortfolioRisk) -> Dict[str, float]:
        """Get risk allocations for different asset classes."""
        return {
            "equity": 0.4,
            "fixed_income": 0.3,
            "commodities": 0.2,
            "cash": 0.1
        }

    def _get_emergency_flags(self, portfolio_risk: PortfolioRisk) -> Dict[str, bool]:
        """Get emergency flags for risk management."""
        return {
            "high_volatility": portfolio_risk.risk_metrics.volatility > 0.5,
            "low_sharpe": portfolio_risk.risk_metrics.sharpe_ratio < 0.5,
            "high_drawdown": portfolio_risk.risk_metrics.max_drawdown < -0.15,
            "negative_var": portfolio_risk.risk_metrics.var_95 < -0.1
        }


# Singleton instance for global use
risk_manager = RiskManager()