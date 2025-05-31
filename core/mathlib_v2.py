from dataclasses import dataclass
import numpy as np
import time

def atr(high, low, close, period=14):
    """Average True Range calculation."""
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

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

def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume Weighted Average Price."""
    return np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0.0

def rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
    rs = gain / loss if loss > 0 else 0.0
    return 100 - (100 / (1 + rs)) if loss > 0 else 100.0

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