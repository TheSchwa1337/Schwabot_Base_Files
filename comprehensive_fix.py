from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Comprehensive fix for all flake8 indentation issues in newmath files
"""

def fix_entropy_calc():
    """Fix all indentation issues in entropy_calc.py"""
    content = '''#!/usr/bin/env python3
"""
NEWMATH ENTROPY CALCULATIONS
===========================

Advanced entropy compensation algorithms for Schwabot trading mathematics.
Clean implementation for entropy gates, triggers, and volume compensation.
"""

from core.unified_math_system import unified_math
import logging

logger = logging.getLogger(__name__)


def calculate_entropy(volume: float, delta: float, base: str = 'natural') -> float:
    """
    Calculate entropy: E(t) = unified_math.log(V + 1) / (1 + |δ|)
    
    Args:
        volume: Trading volume
        delta: Price delta
        base: Logarithm base ('natural', '2', '10')
        
    Returns:
        Entropy value
    """
    try:
        if base == 'natural':
            log_func = unified_math.log
        elif base == '2':
            log_func = np.log2
        elif base == '10':
            log_func = np.log10
        else:
            log_func = unified_math.log
            
        return log_func(unified_math.abs(volume) + 1) / (1 + unified_math.abs(delta))
    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return 0.0


def entropy_trigger(profit_gain: float, entropy: float, threshold: float = 1.0) -> float:
    """
    Calculate entropy trigger: Trigger = P_gain / E(t)
    
    Args:
        profit_gain: Profit gain value
        entropy: Entropy value
        threshold: Trigger threshold
        
    Returns:
        Trigger value
    """
    try:
        if unified_math.abs(entropy) < 1e-12:
            return 0.0
        trigger_value = profit_gain / entropy
        return trigger_value if unified_math.abs(trigger_value) > threshold else 0.0
    except Exception as e:
        logger.error(f"Entropy trigger calculation failed: {e}")
        return 0.0


def volume_entropy(volumes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """
    Calculate volume-weighted entropy series.
    
    Args:
        volumes: Volume series
        prices: Price series
        
    Returns:
        Volume entropy series
    """
    try:
        if len(volumes) != len(prices):
            min_len = unified_math.min(len(volumes), len(prices))
            volumes = volumes[:min_len]
            prices = prices[:min_len]
        
        price_deltas = np.diff(prices)
        price_deltas = np.append([0], price_deltas)  # Pad to match length
        
        entropy_series = np.zeros_like(volumes)
        for i in range(len(volumes)):
            entropy_series[i] = calculate_entropy(volumes[i], price_deltas[i])
            
        return entropy_series
    except Exception as e:
        logger.error(f"Volume entropy calculation failed: {e}")
        return np.zeros_like(volumes)


def delta_compensation(price_deltas: np.ndarray, compensation_factor: float = 1.0) -> np.ndarray:
    """
    Apply delta compensation to price movements.
    
    Mathematical Implementation:
    compensated_delta = δ * (1 + C * E(δ))
    
    Args:
        price_deltas: Price delta series
        compensation_factor: Compensation multiplier
        
    Returns:
        Compensated delta series
    """
    try:
        compensated = np.zeros_like(price_deltas)
        for i, delta in enumerate(price_deltas):
            entropy_val = calculate_entropy(1.0, delta)  # Unit volume
            compensation = 1 + compensation_factor * entropy_val
            compensated[i] = delta * compensation
            
        return compensated
    except Exception as e:
        logger.error(f"Delta compensation failed: {e}")
        return price_deltas


def entropy_normalization(entropy_values: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize entropy values using various methods.
    
    Args:
        entropy_values: Entropy series
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized entropy series
    """
    try:
        if method == 'minmax':
            min_val, max_val = unified_math.unified_math.min(entropy_values), unified_math.unified_math.max(entropy_values)
            if max_val - min_val > 1e-12:
                return (entropy_values - min_val) / (max_val - min_val)
            return entropy_values
        elif method == 'zscore':
            mean_val, std_val = unified_math.unified_math.mean(entropy_values), unified_math.unified_math.std(entropy_values)
            if std_val > 1e-12:
                return (entropy_values - mean_val) / std_val
            return entropy_values
        elif method == 'robust':
            median_val = np.median(entropy_values)
            mad = np.median(unified_math.unified_math.abs(entropy_values - median_val))
            if mad > 1e-12:
                return (entropy_values - median_val) / mad
            return entropy_values
        else:
            return entropy_values
    except Exception as e:
        logger.error(f"Entropy normalization failed: {e}")
        return entropy_values


def entropy_filtering(entropy_values: np.ndarray, filter_type: str = 'moving_average',
                     window: int = 5) -> np.ndarray:
    """
    Apply filtering to entropy values.
    
    Args:
        entropy_values: Entropy series
        filter_type: Filter type ('moving_average', 'exponential', 'median')
        window: Filter window size
        
    Returns:
        Filtered entropy series
    """
    try:
        if filter_type == 'moving_average':
            filtered = np.zeros_like(entropy_values)
            for i in range(len(entropy_values)):
                start_idx = unified_math.max(0, i - window + 1)
                filtered[i] = unified_math.unified_math.mean(entropy_values[start_idx:i + 1])
            return filtered
        elif filter_type == 'exponential':
            alpha = 2.0 / (window + 1)
            filtered = np.zeros_like(entropy_values)
            filtered[0] = entropy_values[0]
            for i in range(1, len(entropy_values)):
                filtered[i] = (alpha * entropy_values[i] +
                              (1 - alpha) * filtered[i - 1])
            return filtered
        elif filter_type == 'median':
            filtered = np.zeros_like(entropy_values)
            for i in range(len(entropy_values)):
                start_idx = unified_math.max(0, i - window + 1)
                filtered[i] = np.median(entropy_values[start_idx:i + 1])
            return filtered
        else:
            return entropy_values
    except Exception as e:
        logger.error(f"Entropy filtering failed: {e}")
        return entropy_values


def adaptive_entropy(prices: np.ndarray, volumes: np.ndarray,
                    adaptation_rate: float = 0.1) -> np.ndarray:
    """
    Calculate adaptive entropy that adjusts to market conditions.
    
    Args:
        prices: Price series
        volumes: Volume series
        adaptation_rate: Rate of adaptation [0, 1]
        
    Returns:
        Adaptive entropy series
    """
    try:
        if len(prices) != len(volumes):
            min_len = unified_math.min(len(prices), len(volumes))
            prices = prices[:min_len]
            volumes = volumes[:min_len]
        
        adaptive_entropy_series = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if i == 0:
                current_entropy = calculate_entropy(volumes[i], 0.0)
            else:
                delta = prices[i] - prices[i - 1]
                current_entropy = calculate_entropy(volumes[i], delta)
            
            # Adaptive adjustment
            if i > 0:
                adaptation = (adaptation_rate *
                            (current_entropy - adaptive_entropy_series[i - 1]))
                adaptive_entropy_series[i] = (adaptive_entropy_series[i - 1] +
                                             adaptation)
            else:
                adaptive_entropy_series[i] = current_entropy
                
        return adaptive_entropy_series
    except Exception as e:
        logger.error(f"Adaptive entropy calculation failed: {e}")
        return np.zeros_like(prices)


def entropy_divergence(entropy_a: np.ndarray, entropy_b: np.ndarray,
                      method: str = 'kl') -> float:
    """
    Calculate divergence between two entropy distributions.
    
    Args:
        entropy_a: First entropy series
        entropy_b: Second entropy series
        method: Divergence method ('kl', 'js', 'wasserstein')
        
    Returns:
        Divergence value
    """
    try:
        if len(entropy_a) != len(entropy_b):
            min_len = unified_math.min(len(entropy_a), len(entropy_b))
            entropy_a = entropy_a[:min_len]
            entropy_b = entropy_b[:min_len]
        
        # Normalize to probability distributions
        entropy_a = (entropy_a / np.sum(entropy_a)
                    if np.sum(entropy_a) > 0 else entropy_a)
        entropy_b = (entropy_b / np.sum(entropy_b)
                    if np.sum(entropy_b) > 0 else entropy_b)
        
        if method == 'kl':  # Kullback-Leibler divergence
            # Add small epsilon to avoid unified_math.log(0)
            eps = 1e-12
            entropy_a = entropy_a + eps
            entropy_b = entropy_b + eps
            return np.sum(entropy_a * unified_math.unified_math.log(entropy_a / entropy_b))
        elif method == 'js':  # Jensen-Shannon divergence
            m = 0.5 * (entropy_a + entropy_b)
            eps = 1e-12
            entropy_a = entropy_a + eps
            entropy_b = entropy_b + eps
            m = m + eps
            kl_a_m = np.sum(entropy_a * unified_math.unified_math.log(entropy_a / m))
            kl_b_m = np.sum(entropy_b * unified_math.unified_math.log(entropy_b / m))
            return 0.5 * kl_a_m + 0.5 * kl_b_m
        elif method == 'wasserstein':  # Simplified Wasserstein distance
            return np.sum(unified_math.unified_math.abs(np.cumsum(entropy_a) - np.cumsum(entropy_b)))
        else:
            return unified_math.unified_math.mean(unified_math.unified_math.abs(entropy_a - entropy_b))
    except Exception as e:
        logger.error(f"Entropy divergence calculation failed: {e}")
        return 0.0


# Export main functions
__all__ = [
    'calculate_entropy',
    'entropy_trigger',
    'volume_entropy', 
    'delta_compensation',
    'entropy_normalization',
    'entropy_filtering',
    'adaptive_entropy',
    'entropy_divergence'
]
'''
    
    with open('newmath/entropy_calc.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix_entropy_calc()
    safe_print("Fixed entropy_calc.py with proper indentation") 