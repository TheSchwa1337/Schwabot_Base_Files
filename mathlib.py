"""
Core Mathematical Library for Quantitative Trading System
This implementation provides a comprehensive set of mathematical tools
for quantitative trading and analysis.
"""

import numpy as np
import pandas as pd
import hashlib
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
try:
    import talib
except ImportError:
    talib = None
from dataclasses import dataclass, field
import logging
from core.risk_manager import RiskManager
from core.risk_monitor import RiskMonitor

class CoreMathLib:
    """
    Core mathematical components for quantitative trading system.
    This implementation extends the original framework with
    additional optimizations and features.
    """
    
    def __init__(self, base_volume: float = 1.0, 
                 tick_freq: float = 1.0, 
                 profit_coef: float = 0.8,
                 threshold: float = 0.5):
        """
        Initialize the mathematical library with core parameters
        
        Args:
            base_volume: Base trading volume
            tick_freq: Frequency of price ticks in Hz
            profit_coef: Profit take coefficient (alpha)
            threshold: Decision threshold for hash-based execution
        """
        # Check if all parameters are valid floats
        if not isinstance(base_volume, float) or not isinstance(tick_freq, float) or \
           not isinstance(profit_coef, float) or not isinstance(threshold, float):
            raise ValueError("All parameters must be of type float.")
        
        # Check if any parameter is off (e.g., less than 0)
        if base_volume < 0 or tick_freq < 0 or profit_coef < 0 or threshold < 0:
            raise ValueError("All parameters must be non-negative.")
        
        self.base_volume = base_volume
        self.tick_freq = tick_freq
        self.delta_t = 1.0 / tick_freq
        self.profit_coef = profit_coef
        self.threshold = threshold
        self.epoch_interval = 3600  # seconds
        
        # Internal state
        self.ema_value = None
        self.last_hash = None
        self.current_prices = []
        self.current_volumes = []
        
    def generate_tick_sequence(self, t0: float, count: int) -> np.ndarray:
        """
        Generate a sequence of tick timestamps
        Enhanced to handle irregular tick intervals if needed
        """
        timestamps = np.array([t0 + n * self.delta_t for n in range(count)])
        return timestamps
    
    def calculate_price_delta(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate the price delta between consecutive ticks
        """
        return np.diff(prices, prepend=prices[0])
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate relative price changes (returns)
        Enhanced to handle edge cases like zero prices
        """
        price_deltas = self.calculate_price_delta(prices)
        prev_prices = np.roll(prices, 1)
        prev_prices[0] = prices[0]
        
        mask = prev_prices != 0
        returns = np.zeros_like(prices, dtype=float)
        returns[mask] = price_deltas[mask] / prev_prices[mask]
        
        return returns
    
    def calculate_profit_ratio(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous profit ratios
        """
        return self.profit_coef * returns
    
    def allocate_volume(self, base_vol: float, n: int, period: int, 
                       method: str = 'sine') -> float:
        """
        Allocate trading volume based on different strategies
        Enhanced with multiple allocation methods
        """
        if method == 'sine':
            beta = 0.2
            return base_vol * (1 + beta * np.sin(2 * np.pi * n / period))
        
        elif method == 'fibonacci':
            fib = [1, 1]
            while len(fib) < period:
                fib.append(fib[-1] + fib[-2])
            return base_vol * fib[min(n, len(fib)-1)] / fib[-1]
        
        elif method == 'gaussian':
            mu = period / 2
            sigma = period / 6
            return base_vol * np.exp(-((n - mu) ** 2) / (2 * sigma ** 2))
        
        else:
            return base_vol
    
    def compute_hash(self, data: Union[str, bytes, List]) -> Tuple[int, float]:
        """
        Compute normalized hash value for decision making
        """
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = str(data).encode('utf-8')
        else:
            data_bytes = data
            
        hash_obj = hashlib.sha256(data_bytes)
        hash_int = int(hash_obj.hexdigest(), 16)
        hash_norm = hash_int / (2 ** 256 - 1)
        
        return hash_int, hash_norm
    
    def hash_decision(self, data: Union[str, bytes, List], threshold: Optional[float] = None) -> bool:
        """
        Make a decision based on hash threshold comparison
        """
        if threshold is None:
            threshold = self.threshold
            
        _, hash_norm = self.compute_hash(data)
        self.last_hash = hash_norm
        
        return hash_norm < threshold
    
    def calculate_hold_fractions(self, profit_ratio: float, mu: float = 0.5) -> Tuple[float, float]:
        """
        Calculate long and short hold fractions
        """
        long_fraction = mu * profit_ratio
        short_fraction = (1 - mu) * profit_ratio
        
        return long_fraction, short_fraction
    
    def detect_drift(self, current_hash: float, drift_threshold: float = 0.01) -> bool:
        """
        Detect drift in hash values
        """
        if self.last_hash is None:
            return False
            
        drift = abs(current_hash - self.last_hash)
        return drift < drift_threshold
    
    def update_ema(self, price: float, alpha: float = 0.2) -> float:
        """
        Update Exponential Moving Average
        """
        if self.ema_value is None:
            self.ema_value = price
        else:
            self.ema_value = alpha * price + (1 - alpha) * self.ema_value
            
        return self.ema_value
    
    def calculate_rolling_std(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Calculate rolling standard deviation
        """
        ema = np.zeros_like(prices)
        for i in range(len(prices)):
            if i == 0:
                ema[i] = prices[i]
            else:
                ema[i] = 0.2 * prices[i] + 0.8 * ema[i-1]
        
        std = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < window:
                window_slice = prices[:i+1]
                ema_val = ema[i]
            else:
                window_slice = prices[i-window+1:i+1]
                ema_val = ema[i]
                
            std[i] = np.sqrt(np.mean((window_slice - ema_val) ** 2))
            
        return std
    
    def calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, k: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Bollinger-style bands
        """
        ema = np.zeros_like(prices)
        for i in range(len(prices)):
            if i == 0:
                ema[i] = prices[i]
            else:
                ema[i] = 0.2 * prices[i] + 0.8 * ema[i-1]
        
        std = self.calculate_rolling_std(prices, window)
        
        upper_band = ema + k * std
        lower_band = ema - k * std
        
        return upper_band, lower_band
    
    def calculate_z_score(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Calculate Z-score (normalized deviation from EMA)
        """
        ema = np.zeros_like(prices)
        for i in range(len(prices)):
            if i == 0:
                ema[i] = prices[i]
            else:
                ema[i] = 0.2 * prices[i] + 0.8 * ema[i-1]
        
        std = self.calculate_rolling_std(prices, window)
        
        z_score = np.zeros_like(prices)
        mask = std != 0
        z_score[mask] = (prices[mask] - ema[mask]) / std[mask]
        
        return z_score
    
    def calculate_risk_adjusted_return(self, returns: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Calculate risk-adjusted returns
        """
        risk_adj = np.zeros_like(returns)
        mask = std != 0
        risk_adj[mask] = returns[mask] / std[mask]
        
        return risk_adj
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, std: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """
        Calculate Sharpe-like ratio
        """
        mean_return = np.mean(returns)
        overall_std = np.mean(std)
        
        if overall_std == 0:
            return 0
            
        return (mean_return - risk_free_rate) / overall_std
    
    def calculate_time_weighted_growth(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate time-weighted growth
        """
        weights_norm = weights / np.sum(weights)
        growth = 1.0
        for r, w in zip(returns, weights_norm):
            growth *= (1 + r) ** w
            
        return growth
    
    def calculate_cumulative_log_return(self, returns: np.ndarray) -> float:
        """
        Calculate cumulative log return
        """
        log_returns = np.log1p(returns)
        return np.sum(log_returns)
    
    def calculate_momentum(self, prices: np.ndarray, lookback: int = 14) -> np.ndarray:
        """
        Calculate momentum signal
        """
        momentum = np.zeros_like(prices)
        for i in range(len(prices)):
            if i >= lookback:
                momentum[i] = prices[i] - prices[i - lookback]
            else:
                momentum[i] = 0
                
        return momentum
    
    def calculate_mean_reversion(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate mean reversion indicator
        """
        ema = np.zeros_like(prices)
        for i in range(len(prices)):
            if i == 0:
                ema[i] = prices[i]
            else:
                ema[i] = 0.2 * prices[i] + 0.8 * ema[i-1]
        
        mean_reversion = ema - prices
        return mean_reversion
    
    def calculate_volume_price_trend(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Calculate Volume-Price Trend
        """
        returns = self.calculate_returns(prices)
        vpt = np.zeros_like(prices)
        for i in range(1, len(prices)):
            vpt[i] = vpt[i-1] + volumes[i] * returns[i]
            
        return vpt
    
    def calculate_price_entropy(self, prices: np.ndarray, bins: int = 10) -> float:
        """
        Calculate entropy of price distribution
        """
        hist, _ = np.histogram(prices, bins=bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return entropy
    
    def tick_bound_execution(self, price: float, lower_bound: float, upper_bound: float, 
                           hash_data: Union[str, bytes, List]) -> bool:
        """
        Decide execution based on price bounds and hash threshold
        """
        in_bounds = lower_bound <= price <= upper_bound
        hash_decision = self.hash_decision(hash_data)
        return in_bounds and hash_decision
    
    def update_adaptive_threshold(self, drift: float, mean_drift: float, gamma: float = 0.1) -> float:
        """
        Update threshold adaptively
        """
        return self.threshold + gamma * (drift - mean_drift)
    
    def check_rebuy_trigger(self, return_val: float, std: float, kappa: float = 0.05, std_threshold: float = 0.02) -> bool:
        """
        Check if rebuy should be triggered
        """
        return (return_val < -kappa) and (std > std_threshold)
    
    def recursive_hash(self, base_hash: bytes, price_delta: float, iterations: int = 1) -> bytes:
        """
        Generate recursive hash sequence
        """
        current_hash = base_hash
        
        for _ in range(iterations):
            combined = current_hash + str(price_delta).encode('utf-8')
            current_hash = hashlib.sha256(combined).digest()
            
        return current_hash
    
    def weighted_sum(self, values: np.ndarray, method: str = 'linear') -> float:
        """
        Calculate weighted sum with different weighting schemes
        """
        n = len(values)
        
        if method == 'linear':
            weights = np.arange(1, n+1)
        elif method == 'quadratic':
            weights = np.arange(1, n+1) ** 2
        elif method == 'exponential':
            weights = np.exp(np.arange(1, n+1))
        else:
            weights = np.ones(n)
            
        weights = weights / np.sum(weights)
        return np.sum(values * weights)
    
    def calculate_cumulative_drift(self, hash_values: np.ndarray) -> float:
        """
        Calculate cumulative drift metric
        """
        drifts = np.abs(np.diff(hash_values, prepend=hash_values[0]))
        return np.sum(drifts)
    
    def calculate_stop_loss(self, price: float, lambda_val: float = 0.1) -> float:
        """
        Calculate stop-loss boundary
        """
        return price * (1 - lambda_val)

    def apply_advanced_strategies(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Apply advanced trading strategies and return comprehensive results
        """
        results = {}
        
        returns = self.calculate_returns(prices)
        results['returns'] = returns
        
        profit_ratios = self.calculate_profit_ratio(returns)
        results['profit_ratios'] = profit_ratios
        
        std = self.calculate_rolling_std(prices)
        results['std'] = std
        
        upper_band, lower_band = self.calculate_bollinger_bands(prices)
        results['upper_band'] = upper_band
        results['lower_band'] = lower_band
        
        z_scores = self.calculate_z_score(prices)
        results['z_scores'] = z_scores
        
        risk_adj_returns = self.calculate_risk_adjusted_return(returns, std)
        results['risk_adj_returns'] = risk_adj_returns
        
        sharpe = self.calculate_sharpe_ratio(returns, std)
        results['sharpe_ratio'] = sharpe
        
        log_return = self.calculate_cumulative_log_return(returns)
        results['log_return'] = log_return
        
        momentum = self.calculate_momentum(prices)
        results['momentum'] = momentum
        
        mean_reversion = self.calculate_mean_reversion(prices)
        results['mean_reversion'] = mean_reversion
        
        vpt = self.calculate_volume_price_trend(prices, volumes)
        results['vpt'] = vpt
        
        entropy = self.calculate_price_entropy(prices)
        results['entropy'] = entropy
        
        return results 

    def calculate_matrix_decay(self) -> float:
        # Calculate matrix decay based on the given parameters
        decay_rate = (self.profit_coef - 1) / self.tick_freq
        return decay_rate * self.base_volume

# Define a function to calculate the topology of a Klein Bottle
def klein_bottle(point):
    """
    Calculate Klein Bottle parametric equations
    Klein bottle: a non-orientable surface in 4D space
    
    Args:
        point: tuple (u, v) where u,v ∈ [0, 2π]
    
    Returns:
        tuple: (x, y, z, w) coordinates in 4D space
    """
    import math
    
    u, v = point
    
    # Klein bottle parametric equations (4D embedding)
    x = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.cos(v)
    y = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.sin(v)
    z = math.sin(v/2) * math.sin(u) + math.cos(v/2) * math.sin(2*u)
    
    # For 3D projection, we can use a simple projection
    w = 0  # Fourth dimension set to 0 for 3D visualization
    
    return (x, y, z, w)

# Define a function to calculate Shannon entropy
def entropy(data):
    """
    Calculate Shannon entropy of data
    
    Args:
        data: array-like data for entropy calculation
    
    Returns:
        float: Shannon entropy value
    """
    import numpy as np
    
    if len(data) == 0:
        return 0.0
    
    # Convert to numpy array if not already
    data = np.array(data)
    
    # Calculate probability distribution
    unique_values, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    
    # Remove zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate Shannon entropy: H = -Σ p(x) * log2(p(x))
    if len(probabilities) == 0:
        return 0.0
    
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy_value

# Define a recursive operation with a depth limit
def recursive_operation(depth_limit, current_depth=0, operation_type='fibonacci'):
    """
    Perform recursive mathematical operation with depth limit
    
    Args:
        depth_limit: target number (for fibonacci) or maximum recursion depth
        current_depth: current recursion depth (for internal use)
        operation_type: type of operation ('fibonacci', 'factorial', 'power')
    
    Returns:
        float: result of recursive operation
    """
    if operation_type == 'fibonacci':
        # Standard fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)
        if depth_limit <= 0:
            return 0.0
        elif depth_limit == 1:
            return 1.0
        else:
            return (recursive_operation(depth_limit - 1, 0, operation_type) + 
                   recursive_operation(depth_limit - 2, 0, operation_type))
    
    elif operation_type == 'factorial':
        # Standard factorial: n! = n * (n-1)!
        if depth_limit <= 1:
            return 1.0
        else:
            return depth_limit * recursive_operation(depth_limit - 1, 0, operation_type)
    
    elif operation_type == 'power':
        # Power of 2: 2^n
        if depth_limit <= 0:
            return 1.0
        else:
            return 2.0 * recursive_operation(depth_limit - 1, 0, operation_type)
    
    else:
        # Default geometric series using current_depth for recursion control
        if current_depth >= depth_limit:
            return 1.0  # Base case
        else:
            return 1.0 + 0.5 * recursive_operation(depth_limit, current_depth + 1, operation_type)

# === MISSING MATHEMATICAL FUNCTIONS FOR MATHLIB_V2 COMPATIBILITY ===
# These functions are imported by mathlib_v2.py but were missing from mathlib.py

def add(a: Union[float, int, np.ndarray], b: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Mathematical addition operation with support for scalars and numpy arrays
    
    Args:
        a: First operand (scalar or array)
        b: Second operand (scalar or array)
    
    Returns:
        Result of a + b (scalar or array)
    """
    return a + b


def subtract(a: Union[float, int, np.ndarray], b: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Mathematical subtraction operation with support for scalars and numpy arrays
    
    Args:
        a: First operand (scalar or array)  
        b: Second operand (scalar or array)
    
    Returns:
        Result of a - b (scalar or array)
    """
    return a - b


# === PROFIT NODE AND TREE CLASSES FOR ADVANCED ANALYTICS ===

@dataclass
class ProfitNode:
    trade_id: str
    pattern_id: str
    parent_id: Optional[str]
    entry_ts: float
    exit_ts: float
    raw_profit: float
    smart_money_score: float
    stop_cause: Optional[str] = None          # from StopLossManager
    lattice_phase: Optional[str] = None       # ALPHA/BETA/…
    children: List["ProfitNode"] = field(default_factory=list)

    # dynamic metrics – filled by ProfitTreeBuilder
    depth: int = 0
    branch_profit: float = 0.0
    fractal_weight: float = 1.0               # from mathlib
    adjusted_profit: float = 0.0              # raw × weight
    path_entropy: float = 0.0                 # Σ|Δ H|


@dataclass
class ProfitTree:
    pattern_id: str
    root_nodes: List[ProfitNode]
    total_profit: float = 0.0
    max_depth: int = 0
    max_width: int = 0
    fractal_kappa: float = 0.0                # global complexity score


# === FEATURE COMPUTATION FOR TRADING SYSTEM ===

def compute_features(prices: List[float], volumes: List[float]) -> Dict[str, float]:
    """
    Compute feature vector for trading decisions
    
    Args:
        prices: List of price values
        volumes: List of volume values
    
    Returns:
        Dictionary with computed features: delta_p, sigma, rsi, confidence
    """
    if len(prices) < 2:
        return {'delta_p': 0.0, 'sigma': 0.0, 'rsi': 50.0, 'confidence': 0.0}
    
    # Price delta
    delta_p = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    
    # Volatility (standard deviation)
    sigma = float(np.std(prices)) if len(prices) > 1 else 0.0
    
    # Simple RSI approximation
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    
    avg_gain = np.mean(gains) if gains else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Confidence based on volume consistency
    volume_stability = 1.0 - (np.std(volumes) / np.mean(volumes)) if len(volumes) > 1 and np.mean(volumes) > 0 else 0.5
    confidence = float(np.clip(volume_stability, 0.0, 1.0))
    
    return {
        'delta_p': float(delta_p),
        'sigma': sigma,
        'rsi': float(rsi),
        'confidence': confidence
    }


# === ERT HELPER CLASS FOR GPU/CPU COMPUTATIONS ===

class ERTHelper:
    """Enhanced Risk Trading Helper for mathematical computations"""
    
    def __init__(self, base_volume: float = 1.0, 
                 tick_freq: float = 1.0, 
                 profit_coef: float = 0.8,
                 threshold: float = 0.5):
        self.math_lib = CoreMathLib(base_volume, tick_freq, profit_coef, threshold)

    def run_ert(self) -> float:
        """Run Enhanced Risk Trading computation"""
        # Perform calculations using CPU
        decay_rate = self.math_lib.calculate_matrix_decay()
        
        # Use CPU computation (GPU computation would require additional libraries)
        result = self.cpu_compute(decay_rate)
        
        return result

    def cpu_compute(self, decay_rate: float) -> float:
        """CPU-based mathematical computation"""
        result = decay_rate * 2.0 + 3.5
        return result

# Create risk manager
risk_manager = RiskManager(
    max_portfolio_risk=0.02,
    max_position_size=0.1,
    max_drawdown=0.15
)

# Create risk monitor
risk_monitor = RiskMonitor(risk_manager)

# Start monitoring
risk_monitor.start_monitoring() 