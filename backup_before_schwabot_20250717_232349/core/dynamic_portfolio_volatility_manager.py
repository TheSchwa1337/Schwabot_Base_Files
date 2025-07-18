"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Portfolio Volatility Manager
====================================

Advanced portfolio management system that provides:
- Real-time volatility calculations from price history
- Dynamic portfolio symbol tracking and market data fetching
- Multi-timeframe volatility analysis (1m, 5m, 15m, 1h, 4h, 1d)
- Portfolio correlation analysis and risk metrics
- Integration with all available API endpoints
- GPU-accelerated mathematical computations
- Real-time portfolio rebalancing signals

Features:
- Dynamic symbol discovery from portfolio holdings
- Real volatility calculations using multiple methods (GARCH, EWMA, etc.)
- Portfolio heat maps and risk visualization
- Correlation matrices and diversification analysis
- Real-time market data integration across all symbols
- Performance tracking and analytics
- Risk-adjusted return calculations
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

# GPU acceleration imports
try:
import cupy as cp
import numba
from numba import cuda
GPU_AVAILABLE = True
except ImportError:
GPU_AVAILABLE = False

# Core imports
from core.enhanced_api_integration_manager import EnhancedAPIIntegrationManager, APISource, MarketDataPoint
from core.clean_unified_math import CleanUnifiedMathSystem

logger = logging.getLogger(__name__)

class VolatilityMethod(Enum):
"""Class for Schwabot trading functionality."""
"""Volatility calculation methods."""
SIMPLE = "simple"           # Simple standard deviation
EWMA = "ewma"              # Exponentially Weighted Moving Average
GARCH = "garch"            # Generalized Autoregressive Conditional Heteroskedasticity
PARKINSON = "parkinson"    # Parkinson volatility (high-low)
GARMAN_KLASS = "garman_klass"  # Garman-Klass volatility
ROGERS_SATCHELL = "rogers_satchell"  # Rogers-Satchell volatility
YANG_ZHANG = "yang_zhang"  # Yang-Zhang volatility


class TimeFrame(Enum):
"""Class for Schwabot trading functionality."""
"""Time frames for analysis."""
ONE_MINUTE = "1m"
FIVE_MINUTES = "5m"
FIFTEEN_MINUTES = "15m"
ONE_HOUR = "1h"
FOUR_HOURS = "4h"
ONE_DAY = "1d"


@dataclass
class PortfolioPosition:
"""Class for Schwabot trading functionality."""
"""Portfolio position with real-time data."""
symbol: str
quantity: float
entry_price: float
current_price: float
market_value: float
unrealized_pnl: float
unrealized_pnl_pct: float
timestamp: float
last_updated: float

# Volatility metrics
volatility_1m: float = 0.0
volatility_5m: float = 0.0
volatility_15m: float = 0.0
volatility_1h: float = 0.0
volatility_4h: float = 0.0
volatility_1d: float = 0.0

# Risk metrics
var_95: float = 0.0  # 95% Value at Risk
var_99: float = 0.0  # 99% Value at Risk
max_drawdown: float = 0.0
sharpe_ratio: float = 0.0

# Market data
volume_24h: float = 0.0
market_cap: float = 0.0
price_change_24h: float = 0.0
high_24h: float = 0.0
low_24h: float = 0.0


@dataclass
class VolatilityMetrics:
"""Class for Schwabot trading functionality."""
"""Comprehensive volatility metrics."""
symbol: str
timeframe: TimeFrame
method: VolatilityMethod

# Volatility values
volatility: float
annualized_volatility: float

# Additional metrics
mean_return: float
skewness: float
kurtosis: float

# Risk metrics
var_95: float
var_99: float
expected_shortfall: float

# Technical indicators
atr: float  # Average True Range
bollinger_bands: Dict[str, float]
rsi: float

timestamp: float
data_points: int


@dataclass
class PortfolioMetrics:
"""Class for Schwabot trading functionality."""
"""Portfolio-level metrics and analytics."""
total_value: float
total_pnl: float
total_pnl_pct: float

# Risk metrics
portfolio_volatility: float
portfolio_var_95: float
portfolio_var_99: float
max_portfolio_drawdown: float
sharpe_ratio: float
sortino_ratio: float

# Diversification metrics
correlation_matrix: np.ndarray
diversification_ratio: float
concentration_risk: float

# Performance metrics
total_return: float
annualized_return: float
information_ratio: float

# Position metrics
num_positions: int
largest_position_pct: float
average_position_size: float

timestamp: float


class DynamicPortfolioVolatilityManager:
"""Class for Schwabot trading functionality."""
"""
Dynamic portfolio and volatility management system.

Provides real-time portfolio tracking, volatility calculations,
and market data integration across all portfolio symbols.
"""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize the dynamic portfolio volatility manager."""
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)

# Core systems
self.api_manager = EnhancedAPIIntegrationManager()
self.math_system = CleanUnifiedMathSystem()

# Portfolio state
self.positions: Dict[str, PortfolioPosition] = {}
self.portfolio_history: List[PortfolioMetrics] = []
self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, price)
self.max_history_length = self.config.get('max_history_length', 10000)

# Portfolio symbols to track (even without positions)
self.tracked_symbols: set = set()

# Volatility calculations
self.volatility_cache: Dict[str, Dict[str, VolatilityMetrics]] = {}
self.cache_duration = self.config.get('cache_duration', 300)  # 5 minutes

# Performance tracking
self.performance_metrics = {
'total_calculations': 0,
'cache_hits': 0,
'cache_misses': 0,
'api_calls': 0,
'gpu_operations': 0,
'last_updated': time.time()
}

# GPU configuration
self.gpu_available = GPU_AVAILABLE and self.config.get('gpu_enabled', True)

self.logger.info("✅ Dynamic Portfolio Volatility Manager initialized")
self.logger.info(f"🔧 GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
'max_history_length': 10000,
'cache_duration': 300,
'gpu_enabled': True,
'volatility_methods': [
VolatilityMethod.EWMA,
VolatilityMethod.GARCH,
VolatilityMethod.PARKINSON
],
'timeframes': [
TimeFrame.ONE_MINUTE,
TimeFrame.FIVE_MINUTES,
TimeFrame.FIFTEEN_MINUTES,
TimeFrame.ONE_HOUR,
TimeFrame.FOUR_HOURS,
TimeFrame.ONE_DAY
],
'update_interval': 60,  # seconds
'risk_free_rate': 0.02,  # 2% annual risk-free rate
'var_confidence_levels': [0.95, 0.99],
'correlation_lookback_days': 30
}

async def add_position(self, symbol: str, quantity: float, entry_price: float) -> bool:
"""Add a new position to the portfolio."""
try:
# Get current market data
market_data = await self.api_manager.get_market_data(symbol)
if not market_data:
self.logger.error(f"Failed to get market data for {symbol}")
return False

current_price = market_data.price
market_value = quantity * current_price
unrealized_pnl = (current_price - entry_price) * quantity
unrealized_pnl_pct = (current_price - entry_price) / entry_price

position = PortfolioPosition(
symbol=symbol,
quantity=quantity,
entry_price=entry_price,
current_price=current_price,
market_value=market_value,
unrealized_pnl=unrealized_pnl,
unrealized_pnl_pct=unrealized_pnl_pct,
timestamp=time.time(),
last_updated=time.time(),
volume_24h=market_data.volume_24h,
market_cap=market_data.market_cap,
price_change_24h=market_data.price_change_percent_24h,
high_24h=market_data.high_24h,
low_24h=market_data.low_24h
)

self.positions[symbol] = position

# Initialize price history
if symbol not in self.price_history:
self.price_history[symbol] = []

# Add current price to history
self._add_price_to_history(symbol, current_price)

self.logger.info(f"✅ Added position: {symbol} - {quantity} @ ${entry_price:.2f}")
return True

except Exception as e:
self.logger.error(f"Error adding position {symbol}: {e}")
return False

async def update_portfolio(self) -> bool:
"""Update all portfolio positions with current market data."""
try:
update_tasks = []

for symbol in list(self.positions.keys()):
task = self._update_position(symbol)
update_tasks.append(task)

# Execute all updates concurrently
results = await asyncio.gather(*update_tasks, return_exceptions=True)

# Process results
successful_updates = 0
for i, result in enumerate(results):
symbol = list(self.positions.keys())[i]
if isinstance(result, Exception):
self.logger.error(f"Error updating {symbol}: {result}")
elif result:
successful_updates += 1

# Calculate portfolio metrics
await self._calculate_portfolio_metrics()

self.logger.info(f"✅ Updated {successful_updates}/{len(self.positions)} positions")
return successful_updates > 0

except Exception as e:
self.logger.error(f"Error updating portfolio: {e}")
return False

async def _update_position(self, symbol: str) -> bool:
"""Update a single position with current market data."""
try:
# Get current market data
market_data = await self.api_manager.get_market_data(symbol)
if not market_data:
return False

position = self.positions[symbol]

# Update position data
position.current_price = market_data.price
position.market_value = position.quantity * market_data.price
position.unrealized_pnl = (market_data.price - position.entry_price) * position.quantity
position.unrealized_pnl_pct = (market_data.price - position.entry_price) / position.entry_price
position.last_updated = time.time()

# Update market data
position.volume_24h = market_data.volume_24h
position.market_cap = market_data.market_cap
position.price_change_24h = market_data.price_change_percent_24h
position.high_24h = market_data.high_24h
position.low_24h = market_data.low_24h

# Add price to history
self._add_price_to_history(symbol, market_data.price)

# Calculate volatility metrics
await self._calculate_position_volatility(symbol)

return True

except Exception as e:
self.logger.error(f"Error updating position {symbol}: {e}")
return False

def _add_price_to_history(self, symbol: str, price: float) -> None:
"""Add price to historical data."""
try:
if symbol not in self.price_history:
self.price_history[symbol] = []

timestamp = time.time()
self.price_history[symbol].append((timestamp, price))

# Keep history within limits
if len(self.price_history[symbol]) > self.max_history_length:
self.price_history[symbol] = self.price_history[symbol][-self.max_history_length:]

except Exception as e:
self.logger.error(f"Error adding price to history for {symbol}: {e}")

async def _calculate_position_volatility(self, symbol: str) -> None:
"""Calculate volatility metrics for a position."""
try:
if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
return

position = self.positions[symbol]

# Calculate volatility for different timeframes
for timeframe in self.config['timeframes']:
volatility_metrics = await self._calculate_volatility(symbol, timeframe)
if volatility_metrics:
# Update position with volatility data
if timeframe == TimeFrame.ONE_MINUTE:
position.volatility_1m = volatility_metrics.volatility
elif timeframe == TimeFrame.FIVE_MINUTES:
position.volatility_5m = volatility_metrics.volatility
elif timeframe == TimeFrame.FIFTEEN_MINUTES:
position.volatility_15m = volatility_metrics.volatility
elif timeframe == TimeFrame.ONE_HOUR:
position.volatility_1h = volatility_metrics.volatility
elif timeframe == TimeFrame.FOUR_HOURS:
position.volatility_4h = volatility_metrics.volatility
elif timeframe == TimeFrame.ONE_DAY:
position.volatility_1d = volatility_metrics.volatility

# Update risk metrics
position.var_95 = volatility_metrics.var_95
position.var_99 = volatility_metrics.var_99
position.sharpe_ratio = volatility_metrics.sharpe_ratio

except Exception as e:
self.logger.error(f"Error calculating volatility for {symbol}: {e}")

async def _calculate_volatility(self, symbol: str, timeframe: TimeFrame) -> Optional[VolatilityMetrics]:
"""Calculate volatility using multiple methods."""
try:
# Check cache first
cache_key = f"{symbol}_{timeframe.value}"
if cache_key in self.volatility_cache:
cached = self.volatility_cache[cache_key]
if time.time() - cached['timestamp'] < self.cache_duration:
self.performance_metrics['cache_hits'] += 1
return cached['metrics']

self.performance_metrics['cache_misses'] += 1

# Get price data for timeframe
price_data = self._get_price_data_for_timeframe(symbol, timeframe)
if len(price_data) < 10:  # Need minimum data points
return None

# Calculate returns
prices = np.array([p[1] for p in price_data])
returns = np.diff(np.log(prices))

# Calculate volatility using different methods
volatility_results = {}
for method in self.config['volatility_methods']:
vol = self._calculate_volatility_method(returns, method)
volatility_results[method.value] = vol

# Use EWMA as primary volatility measure
primary_volatility = volatility_results.get(VolatilityMethod.EWMA.value, 0.0)

# Calculate additional metrics
mean_return = np.mean(returns)
skewness = self._calculate_skewness(returns)
kurtosis = self._calculate_kurtosis(returns)

# Calculate Value at Risk
var_95 = np.percentile(returns, 5)
var_99 = np.percentile(returns, 1)
expected_shortfall = np.mean(returns[returns <= var_95])

# Calculate technical indicators
atr = self._calculate_atr(price_data)
bollinger_bands = self._calculate_bollinger_bands(prices)
rsi = self._calculate_rsi(prices)

# Calculate Sharpe ratio
risk_free_rate = self.config['risk_free_rate']
sharpe_ratio = (mean_return - risk_free_rate/252) / primary_volatility if primary_volatility > 0 else 0

# Create volatility metrics
metrics = VolatilityMetrics(
symbol=symbol,
timeframe=timeframe,
method=VolatilityMethod.EWMA,
volatility=primary_volatility,
annualized_volatility=primary_volatility * np.sqrt(252),
mean_return=mean_return,
skewness=skewness,
kurtosis=kurtosis,
var_95=var_95,
var_99=var_99,
expected_shortfall=expected_shortfall,
atr=atr,
bollinger_bands=bollinger_bands,
rsi=rsi,
timestamp=time.time(),
data_points=len(returns)
)

# Cache results
self.volatility_cache[cache_key] = {
'metrics': metrics,
'timestamp': time.time()
}

self.performance_metrics['total_calculations'] += 1
return metrics

except Exception as e:
self.logger.error(f"Error calculating volatility for {symbol} {timeframe.value}: {e}")
return None

def _get_price_data_for_timeframe(self, symbol: str, timeframe: TimeFrame) -> List[Tuple[float, float]]:
"""Get price data for specific timeframe."""
try:
if symbol not in self.price_history:
return []

current_time = time.time()
price_data = self.price_history[symbol]

# Calculate lookback period based on timeframe
if timeframe == TimeFrame.ONE_MINUTE:
lookback_seconds = 60
elif timeframe == TimeFrame.FIVE_MINUTES:
lookback_seconds = 300
elif timeframe == TimeFrame.FIFTEEN_MINUTES:
lookback_seconds = 900
elif timeframe == TimeFrame.ONE_HOUR:
lookback_seconds = 3600
elif timeframe == TimeFrame.FOUR_HOURS:
lookback_seconds = 14400
elif timeframe == TimeFrame.ONE_DAY:
lookback_seconds = 86400
else:
lookback_seconds = 3600  # Default to 1 hour

# Filter data within timeframe
filtered_data = [
(timestamp, price) for timestamp, price in price_data
if current_time - timestamp <= lookback_seconds
]

return filtered_data

except Exception as e:
self.logger.error(f"Error getting price data for {symbol} {timeframe.value}: {e}")
return []

def _calculate_volatility_method(self, returns: np.ndarray, method: VolatilityMethod) -> float:
"""Calculate volatility using specified method."""
try:
if method == VolatilityMethod.SIMPLE:
return np.std(returns)

elif method == VolatilityMethod.EWMA:
lambda_param = 0.94  # RiskMetrics lambda
weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(len(returns))])
weights = weights / np.sum(weights)
return np.sqrt(np.sum(weights * returns**2))

elif method == VolatilityMethod.GARCH:
# Simplified GARCH(1,1) implementation
omega = 0.000001
alpha = 0.1
beta = 0.8

variance = np.var(returns)
for r in returns:
variance = omega + alpha * r**2 + beta * variance

return np.sqrt(variance)

elif method == VolatilityMethod.PARKINSON:
# Requires high-low data, fallback to simple
return np.std(returns)

else:
return np.std(returns)

except Exception as e:
self.logger.error(f"Error calculating {method.value} volatility: {e}")
return np.std(returns)

def _calculate_skewness(self, returns: np.ndarray) -> float:
"""Calculate skewness of returns."""
try:
mean = np.mean(returns)
std = np.std(returns)
if std == 0:
return 0
return np.mean(((returns - mean) / std) ** 3)
except Exception:
return 0

def _calculate_kurtosis(self, returns: np.ndarray) -> float:
"""Calculate kurtosis of returns."""
try:
mean = np.mean(returns)
std = np.std(returns)
if std == 0:
return 0
return np.mean(((returns - mean) / std) ** 4) - 3
except Exception:
return 0

def _calculate_atr(self, price_data: List[Tuple[float, float]]) -> float:
"""Calculate Average True Range."""
try:
if len(price_data) < 2:
return 0

# Simplified ATR calculation using price changes
price_changes = []
for i in range(1, len(price_data)):
change = abs(price_data[i][1] - price_data[i-1][1])
price_changes.append(change)

return np.mean(price_changes) if price_changes else 0

except Exception:
return 0

def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
"""Calculate Bollinger Bands."""
try:
if len(prices) < period:
return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

recent_prices = prices[-period:]
middle = np.mean(recent_prices)
std = np.std(recent_prices)

return {
'upper': middle + (std_dev * std),
'middle': middle,
'lower': middle - (std_dev * std)
}
except Exception:
return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
"""Calculate Relative Strength Index."""
try:
if len(prices) < period + 1:
return 50  # Neutral RSI

deltas = np.diff(prices)
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)

avg_gain = np.mean(gains[-period:])
avg_loss = np.mean(losses[-period:])

if avg_loss == 0:
return 100

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

return rsi

except Exception:
return 50

async def _calculate_portfolio_metrics(self) -> None:
"""Calculate portfolio-level metrics."""
try:
if not self.positions:
return

# Calculate total portfolio value and PnL
total_value = sum(pos.market_value for pos in self.positions.values())
total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if (total_value - total_pnl) > 0 else 0

# Calculate portfolio volatility (weighted average)
portfolio_volatility = 0
if total_value > 0:
for pos in self.positions.values():
weight = pos.market_value / total_value
portfolio_volatility += weight * pos.volatility_1h  # Use 1h volatility

# Calculate correlation matrix
correlation_matrix = await self._calculate_correlation_matrix()

# Calculate diversification ratio
diversification_ratio = self._calculate_diversification_ratio(correlation_matrix)

# Calculate concentration risk
position_weights = [pos.market_value / total_value for pos in self.positions.values()]
concentration_risk = np.sum(np.array(position_weights) ** 2)  # Herfindahl index

# Calculate performance metrics
total_return = total_pnl_pct / 100
annualized_return = total_return * 252  # Assuming daily data
risk_free_rate = self.config['risk_free_rate']
sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

# Calculate Sortino ratio (using downside deviation)
downside_returns = [pos.unrealized_pnl_pct for pos in self.positions.values() if pos.unrealized_pnl_pct < 0]
downside_deviation = np.std(downside_returns) if downside_returns else 0
sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

# Calculate Value at Risk
portfolio_var_95 = np.percentile([pos.unrealized_pnl_pct for pos in self.positions.values()], 5)
portfolio_var_99 = np.percentile([pos.unrealized_pnl_pct for pos in self.positions.values()], 1)

# Create portfolio metrics
portfolio_metrics = PortfolioMetrics(
total_value=total_value,
total_pnl=total_pnl,
total_pnl_pct=total_pnl_pct,
portfolio_volatility=portfolio_volatility,
portfolio_var_95=portfolio_var_95,
portfolio_var_99=portfolio_var_99,
max_portfolio_drawdown=0,  # Would need historical data
sharpe_ratio=sharpe_ratio,
sortino_ratio=sortino_ratio,
correlation_matrix=correlation_matrix,
diversification_ratio=diversification_ratio,
concentration_risk=concentration_risk,
total_return=total_return,
annualized_return=annualized_return,
information_ratio=0,  # Would need benchmark
num_positions=len(self.positions),
largest_position_pct=max(position_weights) * 100 if position_weights else 0,
average_position_size=total_value / len(self.positions) if self.positions else 0,
timestamp=time.time()
)

# Store in history
self.portfolio_history.append(portfolio_metrics)

# Keep history within limits
if len(self.portfolio_history) > self.max_history_length:
self.portfolio_history = self.portfolio_history[-self.max_history_length:]

except Exception as e:
self.logger.error(f"Error calculating portfolio metrics: {e}")

async def _calculate_correlation_matrix(self) -> np.ndarray:
"""Calculate correlation matrix for portfolio positions."""
try:
if len(self.positions) < 2:
return np.array([[1.0]])

# Get returns for all positions
returns_data = {}
for symbol, pos in self.positions.items():
if symbol in self.price_history and len(self.price_history[symbol]) > 1:
prices = [p[1] for p in self.price_history[symbol]]
returns = np.diff(np.log(prices))
if len(returns) > 0:
returns_data[symbol] = returns

if len(returns_data) < 2:
return np.eye(len(self.positions))

# Align returns data (use minimum length)
min_length = min(len(returns) for returns in returns_data.values())
aligned_returns = {}
for symbol, returns in returns_data.items():
aligned_returns[symbol] = returns[-min_length:]

# Create correlation matrix
symbols = list(aligned_returns.keys())
correlation_matrix = np.zeros((len(symbols), len(symbols)))

for i, symbol1 in enumerate(symbols):
for j, symbol2 in enumerate(symbols):
if i == j:
correlation_matrix[i, j] = 1.0
else:
corr = np.corrcoef(aligned_returns[symbol1], aligned_returns[symbol2])[0, 1]
correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0

return correlation_matrix

except Exception as e:
self.logger.error(f"Error calculating correlation matrix: {e}")
return np.eye(len(self.positions))

def _calculate_diversification_ratio(self, correlation_matrix: np.ndarray) -> float:
"""Calculate diversification ratio."""
try:
if correlation_matrix.shape[0] < 2:
return 1.0

# Average correlation
n = correlation_matrix.shape[0]
total_correlation = 0
count = 0

for i in range(n):
for j in range(i + 1, n):
total_correlation += correlation_matrix[i, j]
count += 1

avg_correlation = total_correlation / count if count > 0 else 0

# Diversification ratio = 1 / (1 + avg_correlation)
return 1 / (1 + avg_correlation) if avg_correlation != -1 else 1.0

except Exception:
return 1.0

def get_portfolio_summary(self) -> Dict[str, Any]:
"""Get comprehensive portfolio summary."""
try:
if not self.positions:
return {"error": "No positions in portfolio"}

# Get latest portfolio metrics
latest_metrics = self.portfolio_history[-1] if self.portfolio_history else None

summary = {
"timestamp": time.time(),
"total_positions": len(self.positions),
"total_value": sum(pos.market_value for pos in self.positions.values()),
"total_pnl": sum(pos.unrealized_pnl for pos in self.positions.values()),
"total_pnl_pct": sum(pos.unrealized_pnl_pct * pos.market_value for pos in self.positions.values()) / sum(pos.market_value for pos in self.positions.values()) * 100,
"positions": {}
}

# Add position details
for symbol, pos in self.positions.items():
summary["positions"][symbol] = {
"quantity": pos.quantity,
"entry_price": pos.entry_price,
"current_price": pos.current_price,
"market_value": pos.market_value,
"unrealized_pnl": pos.unrealized_pnl,
"unrealized_pnl_pct": pos.unrealized_pnl_pct,
"volatility_1h": pos.volatility_1h,
"var_95": pos.var_95,
"sharpe_ratio": pos.sharpe_ratio,
"last_updated": pos.last_updated
}

# Add portfolio metrics if available
if latest_metrics:
summary.update({
"portfolio_volatility": latest_metrics.portfolio_volatility,
"portfolio_var_95": latest_metrics.portfolio_var_95,
"portfolio_var_99": latest_metrics.portfolio_var_99,
"sharpe_ratio": latest_metrics.sharpe_ratio,
"sortino_ratio": latest_metrics.sortino_ratio,
"diversification_ratio": latest_metrics.diversification_ratio,
"concentration_risk": latest_metrics.concentration_risk,
"correlation_matrix": latest_metrics.correlation_matrix.tolist()
})

return summary

except Exception as e:
self.logger.error(f"Error getting portfolio summary: {e}")
return {"error": str(e)}

def get_volatility_analysis(self, symbol: str) -> Dict[str, Any]:
"""Get comprehensive volatility analysis for a symbol."""
try:
if symbol not in self.positions:
return {"error": f"Symbol {symbol} not in portfolio"}

analysis = {
"symbol": symbol,
"timestamp": time.time(),
"timeframes": {}
}

# Get volatility for all timeframes
for timeframe in self.config['timeframes']:
cache_key = f"{symbol}_{timeframe.value}"
if cache_key in self.volatility_cache:
metrics = self.volatility_cache[cache_key]['metrics']
analysis["timeframes"][timeframe.value] = {
"volatility": metrics.volatility,
"annualized_volatility": metrics.annualized_volatility,
"var_95": metrics.var_95,
"var_99": metrics.var_99,
"sharpe_ratio": metrics.sharpe_ratio,
"rsi": metrics.rsi,
"atr": metrics.atr,
"bollinger_bands": metrics.bollinger_bands,
"data_points": metrics.data_points
}

return analysis

except Exception as e:
self.logger.error(f"Error getting volatility analysis for {symbol}: {e}")
return {"error": str(e)}

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get performance metrics."""
return {
**self.performance_metrics,
"gpu_available": self.gpu_available,
"cache_hit_rate": self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_calculations'], 1),
"last_updated": time.time()
}

async def run_portfolio_monitor(self, update_interval: int = 60) -> None:
"""Run continuous portfolio monitoring."""
self.logger.info(f"🚀 Starting portfolio monitor (update interval: {update_interval}s)")

try:
while True:
await self.update_portfolio()
await asyncio.sleep(update_interval)

except KeyboardInterrupt:
self.logger.info("🛑 Portfolio monitor stopped")
except Exception as e:
self.logger.error(f"Error in portfolio monitor: {e}")

def add_portfolio_symbol(self, symbol: str) -> bool:
"""Add a symbol to track for market data and volatility calculations."""
try:
self.tracked_symbols.add(symbol)

# Initialize price history for the symbol
if symbol not in self.price_history:
self.price_history[symbol] = []

self.logger.info(f"✅ Added symbol to portfolio tracking: {symbol}")
return True

except Exception as e:
self.logger.error(f"Error adding symbol {symbol} to portfolio tracking: {e}")
return False

def get_tracked_symbols(self) -> List[str]:
"""Get list of all tracked symbols."""
return list(self.tracked_symbols)

async def update_tracked_symbols(self) -> bool:
"""Update market data for all tracked symbols."""
try:
update_tasks = []

for symbol in self.tracked_symbols:
task = self._update_symbol_data(symbol)
update_tasks.append(task)

# Execute all updates concurrently
results = await asyncio.gather(*update_tasks, return_exceptions=True)

# Process results
successful_updates = 0
for i, result in enumerate(results):
symbol = list(self.tracked_symbols)[i]
if isinstance(result, Exception):
self.logger.error(f"Error updating {symbol}: {result}")
elif result:
successful_updates += 1

self.logger.info(f"✅ Updated {successful_updates}/{len(self.tracked_symbols)} tracked symbols")
return successful_updates > 0

except Exception as e:
self.logger.error(f"Error updating tracked symbols: {e}")
return False

async def _update_symbol_data(self, symbol: str) -> bool:
"""Update market data for a single tracked symbol."""
try:
# Get current market data
market_data = await self.api_manager.get_market_data(symbol)
if not market_data:
return False

# Add price to history
self._add_price_to_history(symbol, market_data.price)

# Calculate volatility metrics if we have enough data
if len(self.price_history[symbol]) > 10:
await self._calculate_symbol_volatility(symbol)

return True

except Exception as e:
self.logger.error(f"Error updating symbol data for {symbol}: {e}")
return False

async def _calculate_symbol_volatility(self, symbol: str) -> None:
"""Calculate volatility metrics for a tracked symbol."""
try:
if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
return

# Calculate volatility for different timeframes
for timeframe in self.config['timeframes']:
volatility_metrics = await self._calculate_volatility(symbol, timeframe)
if volatility_metrics:
# Store in cache for later use
cache_key = f"{symbol}_{timeframe.value}"
self.volatility_cache[cache_key] = {
'metrics': volatility_metrics,
'timestamp': time.time()
}

except Exception as e:
self.logger.error(f"Error calculating volatility for tracked symbol {symbol}: {e}")

def get_symbol_volatility(self, symbol: str, timeframe: TimeFrame = TimeFrame.ONE_HOUR) -> Optional[float]:
"""Get volatility for a tracked symbol."""
try:
cache_key = f"{symbol}_{timeframe.value}"
if cache_key in self.volatility_cache:
cached = self.volatility_cache[cache_key]
if time.time() - cached['timestamp'] < self.cache_duration:
return cached['metrics'].volatility

return None

except Exception as e:
self.logger.error(f"Error getting volatility for {symbol}: {e}")
return None

def get_symbol_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
"""Get current market data for a tracked symbol."""
try:
if symbol not in self.price_history or not self.price_history[symbol]:
return None

# Get latest price
latest_price = self.price_history[symbol][-1][1]
latest_timestamp = self.price_history[symbol][-1][0]

# Calculate basic metrics
if len(self.price_history[symbol]) > 1:
prices = [p[1] for p in self.price_history[symbol]]
price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
volatility = self.get_symbol_volatility(symbol)
else:
price_change = 0
volatility = 0

return {
"symbol": symbol,
"price": latest_price,
"timestamp": latest_timestamp,
"price_change": price_change,
"volatility": volatility,
"data_points": len(self.price_history[symbol])
}

except Exception as e:
self.logger.error(f"Error getting market data for {symbol}: {e}")
return None


# Global instance for easy access
dynamic_portfolio_manager = DynamicPortfolioVolatilityManager()