#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management System for Schwabot Trading Engine
=================================================

Comprehensive risk management with tensor math integration and Schwabot strategy compatibility.
Handles real-time risk assessment, position sizing, and portfolio monitoring.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
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


class ErrorType(Enum):
    """Error types for fault classification."""
    TIMEOUT = "timeout"
    LOGIC_REJECTION = "logic_rejection"
    CCXT_REJECTION = "ccxt_rejection"
    MATHEMATICAL_ERROR = "mathematical_error"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class SafeMode(Enum):
    """Safe mode states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    HALTED = "halted"


@dataclass
class ErrorLogEntry:
    """Error log entry for fault tracking."""
    timestamp: datetime
    error_type: ErrorType
    symbol: Optional[str]
    trade_id: Optional[str]
    error_message: str
    severity: str
    context: Dict[str, Any]
    recovered: bool = False
    recovery_time: Optional[float] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for symbols and system."""
    symbol: str
    error_count: int = 0
    last_error_time: Optional[datetime] = None
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    auto_reset_time: Optional[datetime] = None
    manual_override: bool = False


@dataclass
class RiskMetric:
    """
    Risk metric with tensor math integration and Schwabot compatibility.

    Mathematical Formulas:
    - VaR: VaR = μ - z_α * σ (where z_α is the α-quantile of standard normal)
    - CVaR: CVaR = E[X|X > VaR] (Expected Shortfall)
    - Sharpe: Sharpe = (R_p - R_f) / σ_p (risk-adjusted return)
    - MDD: MDD = max((Peak - Trough) / Peak)
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

    # Class attributes for enum access
    ErrorType = ErrorType
    SafeMode = SafeMode
    RiskLevel = RiskLevel
    ProcessingMode = ProcessingMode

    def __init__(self, risk_tolerance: Union[float, str, Dict[str, Any]] = 0.02, max_portfolio_risk: float = 0.05) -> None:
        """
        Initialize RiskManager with Schwabot-compatible configuration.
        Accepts float, string, or dict for risk_tolerance/config for full dynamic risk integration.
        """
        # Default values
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
        self.strategy_profile = None
        self.risk_config = None

        # 🆕 ROBUST ERROR HANDLING & RECOVERY
        self.error_log: List[ErrorLogEntry] = []
        self.circuit_breaker_states: Dict[str, CircuitBreakerState] = {}
        self.safe_mode = SafeMode.NORMAL
        self.last_error_time: Optional[datetime] = None
        self.error_thresholds = {
            'max_errors_per_symbol': 3,
            'max_errors_per_timeframe': 10,
            'error_timeframe_seconds': 60,
            'circuit_breaker_cooldown_seconds': 300,
            'safe_mode_error_threshold': 5
        }

        # Initialize risk tolerance
        self._initialize_risk_tolerance(risk_tolerance)
        self.max_portfolio_risk = max_portfolio_risk

        logger.info(f"RiskManager initialized with {_backend} backend")

    def _initialize_risk_tolerance(self, risk_tolerance: Union[float, str, Dict[str, Any]]) -> None:
        """Initialize risk tolerance from various input types."""
        if isinstance(risk_tolerance, float):
            self.risk_tolerance = risk_tolerance
        elif isinstance(risk_tolerance, str):
            # Parse risk tolerance from string
            self.risk_tolerance = self._parse_risk_string(risk_tolerance)
        elif isinstance(risk_tolerance, dict):
            # Extract from configuration dictionary
            self.risk_tolerance = risk_tolerance.get('risk_tolerance', 0.02)
            self.risk_config = risk_tolerance
        else:
            self.risk_tolerance = 0.02

    def _parse_risk_string(self, risk_string: str) -> float:
        """Parse risk tolerance from string representation."""
        try:
            # Handle percentage strings
            if '%' in risk_string:
                return float(risk_string.replace('%', '')) / 100
            # Handle decimal strings
            return float(risk_string)
        except ValueError:
            logger.warning(f"Invalid risk string: {risk_string}, using default 0.02")
            return 0.02

    def calculate_var(self, returns: xp.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using tensor operations.

        Mathematical Formula:
        VaR = μ - z_α * σ
        where:
        - μ is the mean return
        - z_α is the α-quantile of standard normal distribution
        - σ is the standard deviation of returns
        """
        try:
            if xp is None or len(returns) == 0:
                return 0.0

            # Calculate mean and standard deviation
            mean_return = xp.mean(returns)
            std_return = xp.std(returns)

            if std_return == 0:
                return 0.0

            # Calculate z-score for confidence level
            z_score = self._get_z_score(confidence_level)

            # Calculate VaR
            var = mean_return - z_score * std_return

            return float(var)

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        # Simplified z-score lookup
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326
        }
        return z_scores.get(confidence_level, 1.645)

    def calculate_cvar(self, returns: xp.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        Mathematical Formula:
        CVaR = E[X|X > VaR]
        """
        try:
            if xp is None or len(returns) == 0:
                return 0.0

            # Calculate VaR first
            var = self.calculate_var(returns, confidence_level)

            # Find returns that exceed VaR
            tail_returns = returns[returns > var]

            if len(tail_returns) == 0:
                return var

            # Calculate expected value of tail returns
            cvar = xp.mean(tail_returns)

            return float(cvar)

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, returns: xp.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted return.

        Mathematical Formula:
        Sharpe = (R_p - R_f) / σ_p
        where:
        - R_p is portfolio return
        - R_f is risk-free rate
        - σ_p is portfolio standard deviation
        """
        try:
            if xp is None or len(returns) == 0:
                return 0.0

            # Calculate mean return and standard deviation
            mean_return = xp.mean(returns)
            std_return = xp.std(returns)

            if std_return == 0:
                return 0.0

            # Calculate Sharpe ratio
            sharpe = (mean_return - risk_free_rate) / std_return

            return float(sharpe)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, prices: xp.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Mathematical Formula:
        MDD = max((Peak - Trough) / Peak)
        """
        try:
            if xp is None or len(prices) < 2:
                return 0.0

            # Calculate cumulative maximum (peak)
            peak = xp.maximum.accumulate(prices)
            
            # Calculate drawdown
            drawdown = (peak - prices) / peak
            
            # Find maximum drawdown
            max_drawdown = xp.max(drawdown)

            return float(max_drawdown)

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_position_size(self, capital: float, risk_per_trade: float, stop_loss_pct: float) -> float:
        """
        Calculate position size using Kelly Criterion and risk management.

        Mathematical Formula:
        f* = (bp - q) / b
        where:
        - f* is the optimal fraction of capital to bet
        - b is the odds received on the bet
        - p is the probability of winning
        - q is the probability of losing (1 - p)
        """
        try:
            if stop_loss_pct <= 0:
                return 0.0

            # Calculate position size based on risk
            position_size = (capital * risk_per_trade) / stop_loss_pct

            # Apply Kelly Criterion if we have win probability
            # For now, use a conservative approach
            kelly_fraction = 0.25  # Conservative Kelly fraction
            position_size *= kelly_fraction

            return float(position_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def assess_portfolio_risk(self, positions: Dict[str, PositionRisk]) -> PortfolioRisk:
        """Assess overall portfolio risk."""
        try:
            if not positions:
                return self._create_empty_portfolio_risk()

            # Calculate portfolio metrics
            total_value = sum(pos.current_value for pos in positions.values())
            total_pnl = sum(pos.unrealized_pnl for pos in positions.values())

            # Calculate portfolio returns for risk metrics
            portfolio_returns = self._calculate_portfolio_returns(positions)

            # Calculate risk metrics
            risk_metrics = RiskMetric(
                var_95=self.calculate_var(portfolio_returns, 0.95),
                var_99=self.calculate_var(portfolio_returns, 0.99),
                cvar_95=self.calculate_cvar(portfolio_returns, 0.95),
                cvar_99=self.calculate_cvar(portfolio_returns, 0.99),
                sharpe_ratio=self.calculate_sharpe_ratio(portfolio_returns),
                sortino_ratio=self.calculate_sharpe_ratio(portfolio_returns),  # Simplified
                max_drawdown=self.calculate_max_drawdown(portfolio_returns),
                volatility=float(xp.std(portfolio_returns)) if xp is not None else 0.0,
                beta=1.0,  # Simplified
                correlation=0.0,  # Simplified
                skewness=0.0,  # Simplified
                kurtosis=0.0,  # Simplified
                risk_hash=self._generate_risk_hash(positions)
            )

            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)

            # Calculate correlation and covariance matrices
            correlation_matrix, covariance_matrix = self._calculate_matrices(positions)

            return PortfolioRisk(
                total_value=total_value,
                total_pnl=total_pnl,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                positions=list(positions.values()),
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix,
                portfolio_hash=self._generate_portfolio_hash(positions)
            )

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return self._create_empty_portfolio_risk()

    def _create_empty_portfolio_risk(self) -> PortfolioRisk:
        """Create empty portfolio risk structure."""
        empty_risk_metrics = RiskMetric(
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
            volatility=0.0, beta=0.0, correlation=0.0, skewness=0.0, kurtosis=0.0
        )

        empty_matrix = xp.zeros((1, 1)) if xp is not None else np.zeros((1, 1))

        return PortfolioRisk(
            total_value=0.0,
            total_pnl=0.0,
            risk_metrics=empty_risk_metrics,
            risk_level=RiskLevel.LOW,
            positions=[],
            correlation_matrix=empty_matrix,
            covariance_matrix=empty_matrix
        )

    def _calculate_portfolio_returns(self, positions: Dict[str, PositionRisk]) -> xp.ndarray:
        """Calculate portfolio returns from positions."""
        try:
            if not positions:
                return xp.array([]) if xp is not None else np.array([])

            # Simplified: use position PnL as returns
            returns = [pos.unrealized_pnl / pos.current_value if pos.current_value > 0 else 0.0 
                      for pos in positions.values()]
            
            return xp.array(returns) if xp is not None else np.array(returns)

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return xp.array([]) if xp is not None else np.array([])

    def _calculate_matrices(self, positions: Dict[str, PositionRisk]) -> Tuple[xp.ndarray, xp.ndarray]:
        """Calculate correlation and covariance matrices."""
        try:
            if not positions:
                empty_matrix = xp.zeros((1, 1)) if xp is not None else np.zeros((1, 1))
                return empty_matrix, empty_matrix

            # Simplified: create identity matrices
            n_positions = len(positions)
            identity_matrix = xp.eye(n_positions) if xp is not None else np.eye(n_positions)
            
            return identity_matrix, identity_matrix

        except Exception as e:
            logger.error(f"Error calculating matrices: {e}")
            empty_matrix = xp.zeros((1, 1)) if xp is not None else np.zeros((1, 1))
            return empty_matrix, empty_matrix

    def _determine_risk_level(self, risk_metrics: RiskMetric) -> RiskLevel:
        """Determine risk level based on metrics."""
        try:
            # Use VaR as primary risk indicator
            var_ratio = abs(risk_metrics.var_95)
            
            if var_ratio <= self.alert_thresholds[RiskLevel.LOW]:
                return RiskLevel.LOW
            elif var_ratio <= self.alert_thresholds[RiskLevel.MEDIUM]:
                return RiskLevel.MEDIUM
            elif var_ratio <= self.alert_thresholds[RiskLevel.HIGH]:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL

        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM

    def _generate_risk_hash(self, positions: Dict[str, PositionRisk]) -> str:
        """Generate hash for risk state tracking."""
        try:
            risk_data = f"{len(positions)}:{sum(pos.risk_metrics.var_95 for pos in positions.values()):.6f}"
            return hashlib.sha256(risk_data.encode()).hexdigest()[:16]
        except Exception:
            return "0000000000000000"

    def _generate_portfolio_hash(self, positions: Dict[str, PositionRisk]) -> str:
        """Generate hash for portfolio state tracking."""
        try:
            portfolio_data = f"{len(positions)}:{sum(pos.current_value for pos in positions.values()):.6f}"
            return hashlib.sha256(portfolio_data.encode()).hexdigest()[:16]
        except Exception:
            return "0000000000000000"

    def log_error(self, error_type: ErrorType, symbol: Optional[str], trade_id: Optional[str], 
                  error_message: str, severity: str = "medium", context: Optional[Dict[str, Any]] = None) -> None:
        """Log error for fault tracking and recovery."""
        try:
            error_entry = ErrorLogEntry(
                timestamp=datetime.now(),
                error_type=error_type,
                symbol=symbol,
                trade_id=trade_id,
                error_message=error_message,
                severity=severity,
                context=context or {}
            )
            
            self.error_log.append(error_entry)
            self.last_error_time = error_entry.timestamp # Update last_error_time
            
            # Update circuit breaker state
            if symbol:
                self._update_circuit_breaker(symbol, error_entry)
            
            # Check for safe mode transition
            self._check_safe_mode_transition()
            
            logger.warning(f"Risk error logged: {error_type.value} for {symbol}: {error_message}")

        except Exception as e:
            logger.error(f"Error logging risk error: {e}")

    def _update_circuit_breaker(self, symbol: str, error_entry: ErrorLogEntry) -> None:
        """Update circuit breaker state for symbol."""
        try:
            if symbol not in self.circuit_breaker_states:
                self.circuit_breaker_states[symbol] = CircuitBreakerState(symbol=symbol)
            
            state = self.circuit_breaker_states[symbol]
            state.error_count += 1
            state.last_error_time = error_entry.timestamp
            
            # Check if circuit breaker should trigger
            if state.error_count >= self.error_thresholds['max_errors_per_symbol']:
                state.triggered = True
                state.trigger_time = error_entry.timestamp
                state.auto_reset_time = datetime.fromtimestamp(
                    error_entry.timestamp.timestamp() + self.error_thresholds['circuit_breaker_cooldown_seconds']
                )
                
                logger.warning(f"Circuit breaker triggered for {symbol}")

        except Exception as e:
            logger.error(f"Error updating circuit breaker: {e}")

    def _check_safe_mode_transition(self) -> None:
        """Check if system should transition to safe mode."""
        try:
            recent_errors = [
                error for error in self.error_log
                if (datetime.now() - error.timestamp).total_seconds() < self.error_thresholds['error_timeframe_seconds']
            ]
            
            if len(recent_errors) >= self.error_thresholds['safe_mode_error_threshold']:
                if self.safe_mode == SafeMode.NORMAL:
                    self.safe_mode = SafeMode.DEGRADED
                    logger.warning("System transitioning to DEGRADED safe mode")
                elif self.safe_mode == SafeMode.DEGRADED:
                    self.safe_mode = SafeMode.EMERGENCY
                    logger.warning("System transitioning to EMERGENCY safe mode")
                elif self.safe_mode == SafeMode.EMERGENCY:
                    self.safe_mode = SafeMode.HALTED
                    logger.critical("System transitioning to HALTED safe mode")

        except Exception as e:
            logger.error(f"Error checking safe mode transition: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'risk_tolerance': self.risk_tolerance,
            'max_portfolio_risk': self.max_portfolio_risk,
            'safe_mode': self.safe_mode.value,
            'processing_mode': self.processing_mode.value,
            'total_errors': len(self.error_log),
            'circuit_breakers_active': sum(1 for cb in self.circuit_breaker_states.values() if cb.triggered),
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'system_health': 'healthy' if len(self.error_log) < 10 else 'degraded' if len(self.error_log) < 50 else 'critical'
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for system monitoring."""
        if not self.error_log:
            return {
                'total_errors': 0,
                'error_types': {},
                'symbols_with_errors': {},
                'recent_errors': [],
                'recovery_rate': 1.0
            }
        
        # Count error types
        error_types = {}
        symbols_with_errors = {}
        recovered_count = 0
        
        for entry in self.error_log:
            # Count error types
            error_type = entry.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count symbols with errors
            if entry.symbol:
                symbols_with_errors[entry.symbol] = symbols_with_errors.get(entry.symbol, 0) + 1
            
            # Count recovered errors
            if entry.recovered:
                recovered_count += 1
        
        # Get recent errors (last 10)
        recent_errors = []
        for entry in self.error_log[-10:]:
            recent_errors.append({
                'timestamp': entry.timestamp.isoformat(),
                'error_type': entry.error_type.value,
                'symbol': entry.symbol,
                'message': entry.error_message,
                'severity': entry.severity,
                'recovered': entry.recovered
            })
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'symbols_with_errors': symbols_with_errors,
            'recent_errors': recent_errors,
            'recovery_rate': recovered_count / len(self.error_log) if self.error_log else 1.0
        }

    def get_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error log entries."""
        if not self.error_log:
            return []
        
        entries = []
        for entry in self.error_log[-limit:]:
            entries.append({
                'timestamp': entry.timestamp.isoformat(),
                'error_type': entry.error_type.value,
                'symbol': entry.symbol,
                'trade_id': entry.trade_id,
                'error_message': entry.error_message,
                'severity': entry.severity,
                'recovered': entry.recovered,
                'recovery_time': entry.recovery_time
            })
        
        return entries

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.positions.clear()
            self.risk_history.clear()
            self.error_log.clear()
            self.circuit_breaker_states.clear()
            logger.info("RiskManager resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up RiskManager: {e}")


# Global instance for easy access
risk_manager = RiskManager()