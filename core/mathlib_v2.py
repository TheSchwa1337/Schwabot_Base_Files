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
import time

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