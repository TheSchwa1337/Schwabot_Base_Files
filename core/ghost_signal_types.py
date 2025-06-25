"""
Ghost Strategy Signal Types - Schwabot UROS v1.0
===============================================

Core data structures for ghost strategy engine with unified math integration.
Provides type-safe ghost signal processing and BTC/USDC volatility analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from numpy.typing import NDArray


@dataclass
class GhostSignal:
    """Individual ghost signal entry with volatility-aware pricing."""
asset: str
price: float
volatility: float
confidence: float
timestamp: float

    def __post_init__(self):
        """Validate and normalize signal data."""
self.price = float(self.price)
        self.volatility = float(self.volatility)
        self.confidence = float(self.confidence)
        self.timestamp = float(self.timestamp)

        # Ensure confidence is bounded
self.confidence = max(0.0, min(1.0, self.confidence))
        # Ensure volatility is non-negative
self.volatility = max(0.0, self.volatility)


# Type alias for ghost array with proper typing
GhostArray = NDArray[np.float64]  # shape: (N, 4) → price, vol, conf, time


@dataclass
class BTCVector:
    """BTC processor vector with ghost array integration."""
ghost_array: GhostArray

    def __post_init__(self):
        """Validate ghost array shape and extract components."""
        if self.ghost_array.shape[1] != 4:
            raise ValueError("GhostArray must have shape (N, 4)")

self.prices = self.ghost_array[:, 0]
self.volatilities = self.ghost_array[:, 1]
self.confidences = self.ghost_array[:, 2]
self.timestamps = self.ghost_array[:, 3]

@property
    def volatility_window(self) -> float:
        """Extract rolling volatility over last 5 entries."""
        if len(self.prices) < 5:
            return 0.0
        return float(np.std(self.prices[-5:]))

@property
    def momentum(self) -> float:
        """Calculate price momentum from differences."""
        if len(self.prices) < 2:
            return 0.0
        return float(np.mean(np.diff(self.prices)))

@property
    def mean_price(self) -> float:
        """Calculate mean price across ghost array."""
        return float(np.mean(self.prices))

@property
    def mean_confidence(self) -> float:
        """Calculate mean confidence across ghost array."""
        return float(np.mean(self.confidences))

    def to_signal(self) -> Dict[str, float]:
        """Convert to unified signal format."""
        return {
"volatility": self.volatility_window,
"momentum": self.momentum,
"mean_price": self.mean_price,
"confidence": self.mean_confidence,
"signal_count": float(len(self.prices))
        }


@dataclass
class GhostStrategyResult:
    """Result from ghost strategy execution."""
strategy_hash: str
action: str
confidence: float
volatility_threshold: float
momentum_threshold: float
execution_ready: bool
signal_data: Dict[str, float]

    def __post_init__(self):
        """Validate result data."""
self.confidence = float(self.confidence)
        self.volatility_threshold = float(self.volatility_threshold)
        self.momentum_threshold = float(self.momentum_threshold)
        self.execution_ready = bool(self.execution_ready)


def build_ghost_array(signals: List[GhostSignal]) -> GhostArray:
    """Convert list of ghost signals to numpy array."""
    if not signals:
        return np.zeros((0, 4), dtype=np.float64)

array_data = [
[s.price, s.volatility, s.confidence, s.timestamp]
        for s in signals
]
    return np.array(array_data, dtype=np.float64)


def extract_volatility_window(ghost_array: GhostArray, window_size: int = 5) -> float:
    """Extract rolling volatility from ghost array."""
    if ghost_array.shape[0] < window_size:
        return 0.0

prices = ghost_array[:, 0]  # BTC/USDC prices
    return float(np.std(prices[-window_size:]))


def validate_ghost_array(ghost_array: GhostArray) -> bool:
    """Validate ghost array structure and data."""
    if ghost_array.ndim != 2 or ghost_array.shape[1] != 4:
        return False

    # Check for valid numeric data
    if not np.all(np.isfinite(ghost_array)):
        return False

    # Check for reasonable price ranges (BTC typically 10k-100k)
    prices = ghost_array[:, 0]
    if np.any(prices < 1000) or np.any(prices > 1000000):
        return False

    # Check for reasonable confidence ranges
confidences = ghost_array[:, 2]
    if np.any(confidences < 0) or np.any(confidences > 1):
        return False

    return True
