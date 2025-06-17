# --- GPU Optional Decorator ---
try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    cp = np
    GPU_ENABLED = False

def gpu_optional(func):
    def wrapper(*args, **kwargs):
        if GPU_ENABLED:
            return func(*args, cp=cp, **kwargs)
        return func(*args, cp=np, **kwargs)
    return wrapper

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import time

# Fix import - import CoreMathLib correctly
try:
    from .mathlib import CoreMathLib
except ImportError:
    from mathlib import CoreMathLib

@dataclass
class SmartStop:
    """Smart stop-loss system with adaptive thresholds."""
    
    initial_stop: float = 0.02  # 2% initial stop
    trailing_distance: float = 0.01  # 1% trailing distance
    max_loss: float = 0.05  # 5% maximum loss
    profit_lock_threshold: float = 0.03  # Lock profits at 3%
    
    def __post_init__(self):
        self.current_stop = self.initial_stop
        self.highest_profit = 0.0
        self.is_active = False
        
    def update(self, current_price: float, entry_price: float) -> dict:
        """Update stop-loss based on current market conditions."""
        profit_pct = (current_price - entry_price) / entry_price
        
        # Track highest profit
        if profit_pct > self.highest_profit:
            self.highest_profit = profit_pct
            
        # Activate trailing stop if profit threshold reached
        if profit_pct >= self.profit_lock_threshold:
            self.is_active = True
            
        # Calculate stop price
        if self.is_active and profit_pct > 0:
            # Trailing stop - follow price up but not down
            trailing_stop = profit_pct - self.trailing_distance
            self.current_stop = max(self.current_stop, trailing_stop)
        else:
            # Fixed stop loss
            self.current_stop = -abs(self.initial_stop)
            
        stop_price = entry_price * (1 + self.current_stop)
        
        return {
            'stop_price': stop_price,
            'current_stop_pct': self.current_stop,
            'profit_pct': profit_pct,
            'is_trailing': self.is_active,
            'should_exit': current_price <= stop_price
        }
    
    def reset(self):
        """Reset the smart stop for a new position."""
        self.current_stop = self.initial_stop
        self.highest_profit = 0.0
        self.is_active = False

class CoreMathLibV2(CoreMathLib):
    """
    Extended mathematical library with v0.2x features
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atr_alpha = 0.1  # ATR smoothing factor
        self.rsi_period = 14  # RSI lookback period
        self.keltner_k = 2.0  # Keltner Channel multiplier
        self.ou_theta = 0.1   # OU mean reversion speed
        self.ou_sigma = 0.1   # OU volatility
        self.memory_lambda = 0.95  # Memory decay factor
        
        # Initialize confidence vector
        self.confidence_vector = np.array([0.5, 0.3, 0.2, 0.1])
        # Initialize weight matrix (learned or decayed)
        self.weight_matrix = np.random.rand(4, 4)
        # Initialize bias vector (learned or adaptive)
        self.bias_vector = np.random.rand(4)
        
    def calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Volume-Weighted Average Price (VWAP)"""
        cumulative_pv = np.cumsum(prices * volumes)
        cumulative_volume = np.cumsum(volumes)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(cumulative_volume != 0,
                          cumulative_pv / cumulative_volume,
                          prices)
        return vwap
    
    def calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range (TR)"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(np.maximum(tr1, tr2), tr3)
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> np.ndarray:
        """Calculate Average True Range (ATR)"""
        tr = self.calculate_true_range(high, low, close)
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        
        for i in range(1, len(tr)):
            atr[i] = self.atr_alpha * tr[i] + (1 - self.atr_alpha) * atr[i-1]
            
        return atr
    
    def calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)"""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initialize first values
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        
        # Calculate smoothed averages
        for i in range(1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (self.rsi_period - 1) + gain[i]) / self.rsi_period
            avg_loss[i] = (avg_loss[i-1] * (self.rsi_period - 1) + loss[i]) / self.rsi_period
        
        # Calculate RS and RSI
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            rsi = 100 - (100 / (1 + rs))
            
        return rsi
    
    def calculate_kelly_fraction(self, returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """Calculate Kelly Criterion Position Fraction"""
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance == 0:
            return 0
            
        return (mean_return - risk_free_rate) / variance
    
    def calculate_risk_parity_weights(self, volatilities: np.ndarray) -> np.ndarray:
        """Calculate risk-parity weights based on inverse volatility"""
        inv_vol = 1 / volatilities
        return inv_vol / np.sum(inv_vol)
    
    def simulate_ornstein_uhlenbeck(self, x0: float, mu: float, n_steps: int) -> np.ndarray:
        """Simulate Ornstein-Uhlenbeck process"""
        dt = 1.0 / self.tick_freq  # Use tick frequency for time step
        x = np.zeros(n_steps)
        x[0] = x0
        
        for i in range(1, n_steps):
            drift = self.ou_theta * (mu - x[i-1]) * dt
            diffusion = self.ou_sigma * np.sqrt(dt) * np.random.normal()
            x[i] = x[i-1] + drift + diffusion
            
        return x
    
    def apply_memory_kernel(self, values: np.ndarray) -> np.ndarray:
        """Apply exponential memory kernel for time decay"""
        n = len(values)
        weights = (1 - self.memory_lambda) * self.memory_lambda ** np.arange(n-1, -1, -1)
        weights = weights / np.sum(weights)
        
        return np.convolve(values, weights, mode='valid')
    
    def apply_advanced_strategies_v2(self, prices: np.ndarray, volumes: np.ndarray,
                                   high: Optional[np.ndarray] = None,
                                   low: Optional[np.ndarray] = None) -> Dict:
        """Apply extended v0.2x trading strategies"""
        # Get base results from parent class
        results = super().apply_advanced_strategies(prices, volumes)
        
        # Calculate VWAP
        results['vwap'] = self.calculate_vwap(prices, volumes)
        
        # Calculate ATR if high/low data is available
        if high is not None and low is not None:
            results['atr'] = self.calculate_atr(high, low, prices)
        
        # Calculate RSI
        results['rsi'] = self.calculate_rsi(prices)
        
        # Calculate Kelly fraction
        returns = np.diff(prices) / prices[:-1]
        results['kelly_fraction'] = self.calculate_kelly_fraction(returns)
        
        # Calculate risk parity weights if we have multiple assets
        if len(prices) > 1:
            volatilities = np.array([np.std(prices)] * 3)  # Mock multiple assets
            results['risk_parity_weights'] = self.calculate_risk_parity_weights(volatilities)
        
        return results

@gpu_optional
def atr(high, low, close, period=14, cp=np):
    """Average True Range calculation."""
    high = cp.asarray(high)
    low = cp.asarray(low)
    close = cp.asarray(close)
    tr = cp.maximum(high[1:] - low[1:], cp.abs(high[1:] - close[:-1]), cp.abs(low[1:] - close[:-1]))
    return float(cp.mean(tr[-period:])) if len(tr) >= period else float(cp.mean(tr))

@dataclass
class PsiStack:
    short: np.ndarray
    mid:   np.ndarray
    long:  np.ndarray
    phantom: np.ndarray

    def collapse(self) -> float:
        """Weighted superposition of all layers."""
        return float(np.mean([np.mean(self.short),
                              np.mean(self.mid),
                              np.mean(self.long),
                              np.mean(self.phantom)]))

def zeta_trigger(delta_mu: float, band: tuple[float, float]) -> bool:
    """Trigger ζ-node if delta_mu is within band."""
    return band[0] <= delta_mu <= band[1]

@gpu_optional
def vwap(prices, volumes, cp=np):
    """Volume Weighted Average Price."""
    prices = cp.asarray(prices)
    volumes = cp.asarray(volumes)
    return float(cp.sum(prices * volumes) / cp.sum(volumes)) if cp.sum(volumes) > 0 else 0.0

@gpu_optional
def rsi(prices, period: int = 14, cp=np):
    """Relative Strength Index."""
    prices = cp.asarray(prices)
    deltas = cp.diff(prices)
    gain = cp.mean(deltas[deltas > 0]) if cp.any(deltas > 0) else 0.0
    loss = -cp.mean(deltas[deltas < 0]) if cp.any(deltas < 0) else 0.0
    rs = gain / loss if loss > 0 else 0.0
    return float(100 - (100 / (1 + rs))) if loss > 0 else 100.0

def klein_bottle_collapse(dim: int = 50) -> np.ndarray:
    """Simulate Klein bottle collapse field."""
    x = np.linspace(-4, 4, dim)
    y = np.linspace(-4, 4, dim)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    phi_t = (time.time() % (2 * np.pi)) * 0.2
    Z = np.cos(2 * np.pi * R - phi_t) * np.exp(-0.5 * R)
    C = np.sin(3*X + phi_t) * np.cos(3*Y - phi_t) * np.exp(-0.3 * R)
    return Z * C

def tesseract_vector(layers: list) -> np.ndarray:
    """Combine 6D command block layers into a single vector."""
    return np.concatenate([np.ravel(l) for l in layers])

def oscillation_phase_wave(tick_index: int, cycle_length: int) -> float:
    """Trigonometric phase oscillation for trajectory scoring."""
    angle = (2 * np.pi / cycle_length) * tick_index
    return abs(np.sin(angle)) 