import numpy as np
from datetime import datetime
import hashlib

def delta_time(t1: datetime, t2: datetime) -> float:
    """Time delta in seconds."""
    return (t2 - t1).total_seconds()

def sum_profit_gradient(profit: np.ndarray, grad: np.ndarray) -> float:
    """Sum of profit weighted by node-gradient."""
    return np.sum(profit * grad)

def calculate_entropy(prices: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of price histogram."""
    hist, _ = np.histogram(prices, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0.0

def kelly_fraction(mu: float, rf: float, sigma2: float) -> float:
    """Kelly criterion for optimal fraction."""
    if sigma2 == 0:
        return 0.0
    return (mu - rf) / sigma2

def sha256_hash(*args) -> str:
    """SHA256 hash of concatenated stringified arguments."""
    s = ''.join(str(a) for a in args)
    return hashlib.sha256(s.encode()).hexdigest()

def combine_hash_logic(hash_block: str, entropy: float, kelly: float) -> str:
    """Combine hash, entropy, and kelly into a new hash block."""
    return sha256_hash(hash_block, entropy, kelly)

def exponential_decay(v0: float, t: float, lambda_: float) -> float:
    """Exponential decay function V(t) = V0 * exp(-lambda * t)."""
    return v0 * np.exp(-lambda_ * t) 