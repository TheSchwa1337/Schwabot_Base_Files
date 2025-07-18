"""
Schwabot Risk Management System with Tensor Math Integration

Advanced risk management with real-time monitoring, fault tolerance, and Schwabot strategy integration.

Features:
- 🆕 Real-time error log accumulator (in-memory + to-disk optional)
- 🆕 Trade-by-trade fault classification
- 🆕 Per-symbol error tracking and circuit breakers
- 🆕 Safe mode with automatic and manual toggles
"""

import hashlib
import json
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
self.error_thresholds = {
'max_errors_per_symbol': 3,
'max_errors_per_timeframe': 10,
'error_timeframe_seconds': 60,
'circuit_breaker_cooldown_seconds': 300,
'safe_mode_error_threshold': 5
}
self.error_stats = {
'total_errors': 0,
'recovered_errors': 0,
'symbol_errors': {},
'error_types': {},
'last_error_time': None
}

# Parse input
if isinstance(risk_tolerance, dict):
self.risk_config = risk_tolerance
self.risk_tolerance = float(risk_tolerance.get('risk_tolerance', 0.02))
self.max_portfolio_risk = float(risk_tolerance.get('max_portfolio_risk', max_portfolio_risk))
self.strategy_profile = risk_tolerance.get('strategy_profile', None)

# 🆕 Load error handling config
error_config = risk_tolerance.get('error_handling', {})
self.error_thresholds.update(error_config)

logger.info(f"🛡️ RiskManager initialized with config dict: {json.dumps(risk_tolerance)}")
elif isinstance(risk_tolerance, str):
# Map string to preset values
mapping = {'low': 0.01, 'medium': 0.03, 'high': 0.05, 'critical': 0.10}
self.risk_tolerance = mapping.get(risk_tolerance.lower(), 0.02)
self.max_portfolio_risk = max_portfolio_risk
logger.info(f"🛡️ RiskManager initialized with string risk_tolerance: {risk_tolerance} -> {self.risk_tolerance:.1%}")
else:
self.risk_tolerance = float(risk_tolerance)
self.max_portfolio_risk = max_portfolio_risk
logger.info(f"🛡️ RiskManager initialized with float risk_tolerance: {self.risk_tolerance:.1%}")

logger.info(f"   Max portfolio risk: {self.max_portfolio_risk:.1%}")
if self.strategy_profile:
logger.info(f"   Strategy profile: {self.strategy_profile}")
if self.risk_config:
logger.info(f"   Full risk config loaded.")

# 🆕 Initialize error recovery system
try:
from core.enhanced_error_recovery_system import EnhancedErrorRecoverySystem
self.error_recovery = EnhancedErrorRecoverySystem()
logger.info("🔄 Enhanced error recovery system integrated")
except ImportError:
self.error_recovery = None
logger.warning("⚠️ Enhanced error recovery system not available")

def log_error(self, error_type: ErrorType, error_message: str, symbol: Optional[str] = None, trade_id: Optional[str] = None, context: Dict[str, Any] = None) -> None:
"""
🆕 Log an error with comprehensive tracking.

Args:
error_type: Type of error for classification
error_message: Human-readable error message
symbol: Trading symbol (if applicable)
trade_id: Trade ID (if applicable)
context: Additional context information
"""
try:
# Create error log entry
error_entry = ErrorLogEntry(
timestamp=datetime.now(),
error_type=error_type,
symbol=symbol,
trade_id=trade_id,
error_message=error_message,
severity=self._determine_error_severity(error_type),
context=context or {}
)

# Add to error log
self.error_log.append(error_entry)

# Update error statistics
self.error_stats['total_errors'] += 1
self.error_stats['last_error_time'] = datetime.now()

# Update symbol error tracking
if symbol:
if symbol not in self.error_stats['symbol_errors']:
self.error_stats['symbol_errors'][symbol] = 0
self.error_stats['symbol_errors'][symbol] += 1

# Check circuit breaker for symbol
self._check_symbol_circuit_breaker(symbol)

# Update error type tracking
error_type_str = error_type.value
if error_type_str not in self.error_stats['error_types']:
self.error_stats['error_types'][error_type_str] = 0
self.error_stats['error_types'][error_type_str] += 1

# Check system-wide circuit breakers
self._check_system_circuit_breakers()

# Log the error
logger.error(f"🚨 Error logged: {error_type.value} - {error_message} (Symbol: {symbol}, Trade: {trade_id})")

except Exception as e:
logger.error(f"❌ Failed to log error: {e}")

def _determine_error_severity(self, error_type: ErrorType) -> str:
"""Determine error severity based on type."""
severity_map = {
ErrorType.TIMEOUT: "medium",
ErrorType.LOGIC_REJECTION: "low",
ErrorType.CCXT_REJECTION: "high",
ErrorType.MATHEMATICAL_ERROR: "high",
ErrorType.NETWORK_ERROR: "medium",
ErrorType.MEMORY_ERROR: "critical",
ErrorType.SYSTEM_ERROR: "critical",
ErrorType.UNKNOWN: "medium"
}
return severity_map.get(error_type, "medium")

def _check_symbol_circuit_breaker(self, symbol: str) -> None:
"""Check and trigger circuit breaker for specific symbol."""
try:
# Initialize circuit breaker state if not exists
if symbol not in self.circuit_breaker_states:
self.circuit_breaker_states[symbol] = CircuitBreakerState(symbol=symbol)

state = self.circuit_breaker_states[symbol]
current_time = datetime.now()

# Reset error count if outside timeframe
if (state.last_error_time and
(current_time - state.last_error_time).total_seconds() > self.error_thresholds['error_timeframe_seconds']):
state.error_count = 0

# Update error count and time
state.error_count += 1
state.last_error_time = current_time

# Check if circuit breaker should trigger
if (state.error_count >= self.error_thresholds['max_errors_per_symbol'] and
not state.triggered and not state.manual_override):

state.triggered = True
state.trigger_time = current_time
state.auto_reset_time = datetime.fromtimestamp(
current_time.timestamp() + self.error_thresholds['circuit_breaker_cooldown_seconds']
)

logger.critical(f"🚨 Circuit breaker triggered for {symbol} after {state.error_count} errors")

except Exception as e:
logger.error(f"❌ Failed to check symbol circuit breaker: {e}")

def _check_system_circuit_breakers(self) -> None:
"""Check system-wide circuit breakers and safe mode."""
try:
current_time = datetime.now()

# Check if too many errors in timeframe
recent_errors = [
error for error in self.error_log
if (current_time - error.timestamp).total_seconds() <= self.error_thresholds['error_timeframe_seconds']
]

if len(recent_errors) >= self.error_thresholds['safe_mode_error_threshold']:
self._enter_safe_mode(SafeMode.EMERGENCY, f"Too many errors: {len(recent_errors)} in {self.error_thresholds['error_timeframe_seconds']}s")

# Check if any symbol has too many errors
for symbol, error_count in self.error_stats['symbol_errors'].items():
if error_count >= self.error_thresholds['max_errors_per_symbol']:
logger.warning(f"⚠️ Symbol {symbol} has {error_count} errors - consider manual intervention")

except Exception as e:
logger.error(f"❌ Failed to check system circuit breakers: {e}")

def _enter_safe_mode(self, mode: SafeMode, reason: str) -> None:
"""Enter safe mode with specified level."""
try:
if self.safe_mode != mode:
old_mode = self.safe_mode
self.safe_mode = mode
logger.critical(f"🚨 Safe mode changed: {old_mode.value} -> {mode.value} (Reason: {reason})")

# Log the safe mode change
self.log_error(
ErrorType.SYSTEM_ERROR,
f"Safe mode entered: {mode.value} - {reason}",
context={'old_mode': old_mode.value, 'new_mode': mode.value}
)

except Exception as e:
logger.error(f"❌ Failed to enter safe mode: {e}")

def exit_safe_mode(self, reason: str = "Manual override") -> None:
"""Exit safe mode."""
try:
if self.safe_mode != SafeMode.NORMAL:
old_mode = self.safe_mode
self.safe_mode = SafeMode.NORMAL
logger.info(f"✅ Safe mode exited: {old_mode.value} -> normal (Reason: {reason})")

# Log the safe mode exit
self.log_error(
ErrorType.SYSTEM_ERROR,
f"Safe mode exited: {old_mode.value} -> normal - {reason}",
context={'old_mode': old_mode.value, 'new_mode': 'normal'}
)

except Exception as e:
logger.error(f"❌ Failed to exit safe mode: {e}")

def reset_circuit_breaker(self, symbol: str, manual: bool = False) -> bool:
"""Reset circuit breaker for a symbol."""
try:
if symbol in self.circuit_breaker_states:
state = self.circuit_breaker_states[symbol]
state.triggered = False
state.error_count = 0
state.trigger_time = None
state.auto_reset_time = None
state.manual_override = manual

logger.info(f"✅ Circuit breaker reset for {symbol} ({'manual' if manual else 'automatic'})")
return True
return False
except Exception as e:
logger.error(f"❌ Failed to reset circuit breaker: {e}")
return False

def get_error_log(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
"""Get error log entries."""
try:
entries = self.error_log
if symbol:
entries = [e for e in entries if e.symbol == symbol]

# Return most recent entries
recent_entries = entries[-limit:] if len(entries) > limit else entries

return [
{
'timestamp': entry.timestamp.isoformat(),
'error_type': entry.error_type.value,
'symbol': entry.symbol,
'trade_id': entry.trade_id,
'error_message': entry.error_message,
'severity': entry.severity,
'context': entry.context,
'recovered': entry.recovered,
'recovery_time': entry.recovery_time
}
for entry in recent_entries
]

except Exception as e:
logger.error(f"❌ Failed to get error log: {e}")
return []

def get_error_statistics(self) -> Dict[str, Any]:
"""Get comprehensive error statistics."""
try:
current_time = datetime.now()

# Calculate recent error rates
recent_errors = [
error for error in self.error_log
if (current_time - error.timestamp).total_seconds() <= self.error_thresholds['error_timeframe_seconds']
]

# Calculate recovery rate
total_errors = self.error_stats['total_errors']
recovered_errors = self.error_stats['recovered_errors']
recovery_rate = (recovered_errors / total_errors * 100) if total_errors > 0 else 0

return {
'total_errors': total_errors,
'recovered_errors': recovered_errors,
'recovery_rate_percent': recovery_rate,
'recent_errors': len(recent_errors),
'symbol_errors': self.error_stats['symbol_errors'],
'error_types': self.error_stats['error_types'],
'safe_mode': self.safe_mode.value,
'circuit_breakers_triggered': sum(1 for state in self.circuit_breaker_states.values() if state.triggered),
'last_error_time': self.error_stats['last_error_time'].isoformat() if self.error_stats['last_error_time'] else None
}

except Exception as e:
logger.error(f"❌ Failed to get error statistics: {e}")
return {}

def can_trade_symbol(self, symbol: str) -> bool:
"""Check if trading is allowed for a specific symbol."""
try:
# Check safe mode
if self.safe_mode == SafeMode.HALTED:
return False

# Check symbol circuit breaker
if symbol in self.circuit_breaker_states:
state = self.circuit_breaker_states[symbol]
if state.triggered:
# Check if auto-reset time has passed
if state.auto_reset_time and datetime.now() >= state.auto_reset_time:
self.reset_circuit_breaker(symbol, manual=False)
else:
return False

return True

except Exception as e:
logger.error(f"❌ Failed to check symbol trading status: {e}")
return False

def get_system_status(self) -> Dict[str, Any]:
"""Get comprehensive system status including error handling."""
try:
risk_summary = self.get_risk_summary()
error_stats = self.get_error_statistics()

return {
'risk_management': risk_summary,
'error_handling': error_stats,
'safe_mode': self.safe_mode.value,
'processing_mode': self.processing_mode.value,
'circuit_breakers': {
symbol: {
'triggered': state.triggered,
'error_count': state.error_count,
'last_error_time': state.last_error_time.isoformat() if state.last_error_time else None,
'auto_reset_time': state.auto_reset_time.isoformat() if state.auto_reset_time else None
}
for symbol, state in self.circuit_breaker_states.items()
},
'timestamp': datetime.now().isoformat()
}

except Exception as e:
logger.error(f"❌ Failed to get system status: {e}")
return {'error': str(e)}

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
# 🆕 Log mathematical error
self.log_error(
ErrorType.MATHEMATICAL_ERROR,
f"VaR calculation failed: {e}",
context={'confidence_level': confidence_level, 'returns_length': len(returns)}
)
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
# 🆕 Log mathematical error
self.log_error(
ErrorType.MATHEMATICAL_ERROR,
f"Expected Shortfall calculation failed: {e}",
context={'confidence_level': confidence_level, 'returns_length': len(returns)}
)
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
# 🆕 Log mathematical error
self.log_error(
ErrorType.MATHEMATICAL_ERROR,
f"Sharpe ratio calculation failed: {e}",
context={'risk_free_rate': risk_free_rate, 'returns_length': len(returns)}
)
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
# 🆕 Log mathematical error
self.log_error(
ErrorType.MATHEMATICAL_ERROR,
f"Maximum drawdown calculation failed: {e}",
context={'returns_length': len(returns)}
)
logger.error(f"❌ Failed to compute maximum drawdown: {e}")
raise

def calculate_risk_metrics(self, returns: xp.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> RiskMetric:
"""
Calculate comprehensive risk metrics using tensor operations with robust edge case handling.

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
# Robust input validation
if xp is None:
raise ValueError("Tensor operations not available")

# Handle None input
if returns is None:
logger.warning("Returns array is None, using empty array")
returns = xp.array([])

# Convert to numpy array if needed
if not isinstance(returns, xp.ndarray):
try:
returns = xp.array(returns)
except Exception as e:
logger.error(f"Failed to convert returns to array: {e}")
raise ValueError(f"Invalid returns data type: {type(returns)}")

# Handle empty array with safe defaults
if len(returns) == 0:
logger.warning("Returns array is empty, returning default risk metrics")
return self._get_default_risk_metrics()

# Handle single value array
if len(returns) == 1:
logger.warning("Returns array has only one value, using conservative estimates")
return self._get_single_value_risk_metrics(returns[0])

# Validate data quality
if not xp.all(xp.isfinite(returns)):
logger.warning("Returns array contains NaN or infinite values, cleaning data")
returns = self._clean_returns_data(returns)
if len(returns) == 0:
return self._get_default_risk_metrics()

# Basic statistics with error handling
try:
mean_return = float(xp.mean(returns))
volatility = float(xp.std(returns))
except Exception as e:
logger.error(f"Failed to calculate basic statistics: {e}")
return self._get_default_risk_metrics()

# Value at Risk (VaR) - using individual function with error handling
try:
var_95 = self.compute_var(returns, 0.95)
var_99 = self.compute_var(returns, 0.99)
except Exception as e:
logger.warning(f"VaR calculation failed: {e}, using defaults")
var_95 = -0.02  # Conservative default
var_99 = -0.03  # Conservative default

# Expected Shortfall (ES) - using individual function with error handling
try:
cvar_95 = self.compute_expected_shortfall(returns, 0.95)
cvar_99 = self.compute_expected_shortfall(returns, 0.99)
except Exception as e:
logger.warning(f"Expected Shortfall calculation failed: {e}, using defaults")
cvar_95 = -0.025  # Conservative default
cvar_99 = -0.035  # Conservative default

# Sharpe and Sortino ratios with error handling
try:
risk_free_rate = 0.02  # 2% annual risk-free rate
sharpe_ratio = self.compute_sharpe_ratio(returns, risk_free_rate)

# Sortino ratio (downside deviation)
downside_returns = returns[returns < mean_return]
if len(downside_returns) > 0:
downside_deviation = float(xp.std(downside_returns) * xp.sqrt(252))
sortino_ratio = (mean_return * 252 - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
else:
sortino_ratio = 0.0
except Exception as e:
logger.warning(f"Sharpe/Sortino calculation failed: {e}, using defaults")
sharpe_ratio = 0.0
sortino_ratio = 0.0

# Maximum drawdown - using individual function with error handling
try:
max_drawdown = self.compute_max_drawdown(returns)
except Exception as e:
logger.warning(f"Max drawdown calculation failed: {e}, using default")
max_drawdown = -0.01  # Conservative default

# Higher moments with error handling
try:
if volatility > 0:
standardized_returns = (returns - mean_return) / volatility
skewness = float(xp.mean(standardized_returns ** 3))
kurtosis = float(xp.mean(standardized_returns ** 4))
else:
skewness = 0.0
kurtosis = 0.0
except Exception as e:
logger.warning(f"Higher moments calculation failed: {e}, using defaults")
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
# 🆕 Log mathematical error
self.log_error(
ErrorType.MATHEMATICAL_ERROR,
f"Risk metrics calculation failed: {e}",
context={'confidence_levels': confidence_levels, 'returns_length': len(returns) if returns is not None else 0}
)
logger.error(f"❌ Failed to calculate risk metrics: {e}")
return self._get_default_risk_metrics()

def _get_default_risk_metrics(self) -> RiskMetric:
"""Get default risk metrics for edge cases."""
return RiskMetric(
var_95=-0.02, var_99=-0.03, cvar_95=-0.025, cvar_99=-0.035,
sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=-0.01,
volatility=0.01, beta=1.0, correlation=0.0,
skewness=0.0, kurtosis=0.0, tensor_confidence=0.0,
risk_hash="default_risk_hash"
)

def _get_single_value_risk_metrics(self, single_return: float) -> RiskMetric:
"""Get risk metrics for single return value."""
# Conservative estimates for single value
volatility = abs(single_return) if xp.isfinite(single_return) else 0.01
var_95 = -volatility * 1.645  # 95% confidence interval
var_99 = -volatility * 2.326  # 99% confidence interval

return RiskMetric(
var_95=var_95, var_99=var_99, cvar_95=var_95, cvar_99=var_99,
sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=min(0, single_return),
volatility=volatility, beta=1.0, correlation=0.0,
skewness=0.0, kurtosis=0.0, tensor_confidence=0.1,
risk_hash="single_value_hash"
)

def _clean_returns_data(self, returns: xp.ndarray) -> xp.ndarray:
"""Clean returns data by removing NaN and infinite values."""
try:
# Remove NaN and infinite values
clean_returns = returns[xp.isfinite(returns)]

if len(clean_returns) == 0:
logger.warning("No valid data after cleaning")
return xp.array([])

logger.info(f"Cleaned returns data: {len(returns)} -> {len(clean_returns)} valid values")
return clean_returns

except Exception as e:
logger.error(f"Failed to clean returns data: {e}")
return xp.array([])

def assess_position_risk(self, symbol: str, position_size: float, current_price: float, historical_returns: xp.ndarray, entry_price: float) -> PositionRisk:
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
# 🆕 Check if symbol is allowed to trade
if not self.can_trade_symbol(symbol):
raise ValueError(f"Trading not allowed for {symbol} due to circuit breaker or safe mode")

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
# 🆕 Log error with symbol context
self.log_error(
ErrorType.LOGIC_REJECTION,
f"Position risk assessment failed for {symbol}: {e}",
symbol=symbol,
context={'position_size': position_size, 'current_price': current_price, 'entry_price': entry_price}
)
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
# 🆕 Log error
self.log_error(
ErrorType.LOGIC_REJECTION,
f"Portfolio risk assessment failed: {e}",
context={'portfolio_data_keys': list(portfolio_data.keys())}
)
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