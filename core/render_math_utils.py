"""
Render Mathematical Utilities
=============================

Mathematical functions for dynamic line rendering including scoring algorithms,
entropy-based styling, time decay effects, and resource-aware adjustments.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def calculate_line_score(profit: float, entropy: float, 
                        weight_profit: float = 0.7, weight_entropy: float = 0.3) -> float:
    """
    Calculate a score for a line based on profit and entropy.
    Uses tanh for profit to bound its effect and ensures entropy reduces the score.

    Args:
        profit: The profit associated with the line
        entropy: The entropy or unpredictability associated with the line
        weight_profit: Weight for the profit component (default: 0.7)
        weight_entropy: Weight for the entropy component (default: 0.3)

    Returns:
        A calculated score for the line, typically in range [-1, 1]
    """
    try:
        # Bound profit using tanh to prevent extreme values
        profit_component = weight_profit * np.tanh(profit)
        
        # Entropy reduces score (higher entropy = lower score)
        entropy_component = weight_entropy * entropy
        
        score = profit_component - entropy_component
        
        logger.debug(f"Line score calculated: profit={profit:.3f}, entropy={entropy:.3f}, score={score:.3f}")
        return float(score)
        
    except Exception as e:
        logger.error(f"Error calculating line score: {e}")
        return 0.0

def determine_line_style(entropy: float, entropy_threshold: float = 0.5) -> str:
    """
    Determine the visual style of a line based on its entropy.

    Args:
        entropy: The entropy value of the line
        entropy_threshold: The threshold above which the line style changes

    Returns:
        'solid' if entropy is below the threshold, 'dashed' otherwise
    """
    try:
        if entropy < entropy_threshold:
            return 'solid'
        elif entropy < entropy_threshold * 1.5:
            return 'dashed'
        else:
            return 'dotted'
    except Exception as e:
        logger.error(f"Error determining line style: {e}")
        return 'solid'

def calculate_decay(last_update: datetime, half_life_seconds: int = 3600) -> float:
    """
    Calculate a decay factor based on the age of the line.
    Simulates a half-life decay, where the factor reduces by half every `half_life_seconds`.

    Args:
        last_update: The datetime when the line was last updated
        half_life_seconds: The time in seconds for the line's "value" to halve

    Returns:
        A decay factor between 0 and 1, where 1 means no decay
    """
    try:
        elapsed_seconds = (datetime.now() - last_update).total_seconds()
        
        # Handle future dates or negative elapsed time
        if elapsed_seconds <= 0:
            return 1.0
            
        # Decay formula: e^(-lambda * t), where lambda = ln(2) / half_life
        decay_constant = np.log(2) / half_life_seconds
        decay_factor = np.exp(-decay_constant * elapsed_seconds)
        
        # Ensure decay factor is between 0 and 1
        decay_factor = max(0.0, min(1.0, decay_factor))
        
        logger.debug(f"Decay calculated: elapsed={elapsed_seconds:.1f}s, factor={decay_factor:.3f}")
        return float(decay_factor)
        
    except Exception as e:
        logger.error(f"Error calculating decay: {e}")
        return 1.0

def adjust_line_thickness(base_thickness: int, memory_usage_pct: float, 
                         cpu_usage_pct: Optional[float] = None) -> int:
    """
    Adjust line thickness based on system resource usage.
    Reduces thickness if memory or CPU usage is high to indicate system load.

    Args:
        base_thickness: The default or desired line thickness
        memory_usage_pct: Current system memory usage as a percentage (0-100)
        cpu_usage_pct: Optional CPU usage percentage for additional adjustment

    Returns:
        The adjusted line thickness (minimum 1)
    """
    try:
        adjusted_thickness = base_thickness
        
        # Adjust based on memory usage
        if memory_usage_pct > 90:
            adjusted_thickness = max(1, int(base_thickness * 0.3))
        elif memory_usage_pct > 80:
            adjusted_thickness = max(1, int(base_thickness * 0.5))
        elif memory_usage_pct > 70:
            adjusted_thickness = max(1, int(base_thickness * 0.7))
            
        # Additional adjustment based on CPU usage if provided
        if cpu_usage_pct is not None:
            if cpu_usage_pct > 90:
                adjusted_thickness = max(1, int(adjusted_thickness * 0.5))
            elif cpu_usage_pct > 80:
                adjusted_thickness = max(1, int(adjusted_thickness * 0.7))
                
        logger.debug(f"Thickness adjusted: base={base_thickness}, mem={memory_usage_pct:.1f}%, "
                    f"cpu={cpu_usage_pct or 'N/A'}%, adjusted={adjusted_thickness}")
        
        return adjusted_thickness
        
    except Exception as e:
        logger.error(f"Error adjusting line thickness: {e}")
        return max(1, base_thickness)

def calculate_line_opacity(decay_factor: float, confidence: float = 1.0, 
                          min_opacity: float = 0.1) -> float:
    """
    Calculate line opacity based on decay factor and confidence.
    
    Args:
        decay_factor: Time-based decay factor (0-1)
        confidence: Confidence in the line data (0-1)
        min_opacity: Minimum opacity to maintain visibility
        
    Returns:
        Opacity value between min_opacity and 1.0
    """
    try:
        # Combine decay and confidence
        opacity = decay_factor * confidence
        
        # Ensure minimum visibility
        opacity = max(min_opacity, min(1.0, opacity))
        
        return float(opacity)
        
    except Exception as e:
        logger.error(f"Error calculating opacity: {e}")
        return min_opacity

def determine_line_color(score: float, entropy: float) -> str:
    """
    Determine line color based on score and entropy.
    
    Args:
        score: Line score (-1 to 1)
        entropy: Entropy value (0 to 1+)
        
    Returns:
        Hex color code string
    """
    try:
        # Base color on score
        if score > 0.5:
            base_color = "#00FF00"  # Green for high positive score
        elif score > 0:
            base_color = "#FFFF00"  # Yellow for moderate positive score
        elif score > -0.5:
            base_color = "#FFA500"  # Orange for moderate negative score
        else:
            base_color = "#FF0000"  # Red for high negative score
            
        # Modify saturation based on entropy
        if entropy > 0.8:
            # High entropy - desaturate color
            base_color = "#808080"  # Gray for high uncertainty
            
        return base_color
        
    except Exception as e:
        logger.error(f"Error determining line color: {e}")
        return "#FFFFFF"  # Default white

def calculate_volatility_score(price_data: List[float], window: int = 20) -> float:
    """
    Calculate volatility score from price data.
    
    Args:
        price_data: List of price values
        window: Rolling window size for volatility calculation
        
    Returns:
        Volatility score (standard deviation of returns)
    """
    try:
        if len(price_data) < 2:
            return 0.0
            
        # Calculate returns
        prices = np.array(price_data)
        returns = np.diff(prices) / prices[:-1]
        
        # Use rolling window if data is long enough
        if len(returns) >= window:
            returns = returns[-window:]
            
        volatility = np.std(returns)
        
        logger.debug(f"Volatility calculated: {volatility:.4f} from {len(price_data)} prices")
        return float(volatility)
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0

def smooth_line_path(path: List[float], smoothing_factor: float = 0.1) -> List[float]:
    """
    Apply exponential smoothing to a line path.
    
    Args:
        path: List of path values
        smoothing_factor: Smoothing factor (0-1, lower = more smoothing)
        
    Returns:
        Smoothed path values
    """
    try:
        if len(path) <= 1:
            return path
            
        smoothed = [path[0]]  # Start with first value
        
        for i in range(1, len(path)):
            # Exponential smoothing formula
            smoothed_value = (smoothing_factor * path[i] + 
                            (1 - smoothing_factor) * smoothed[i-1])
            smoothed.append(smoothed_value)
            
        return smoothed
        
    except Exception as e:
        logger.error(f"Error smoothing line path: {e}")
        return path

def calculate_trend_strength(values: List[float]) -> Tuple[float, str]:
    """
    Calculate trend strength and direction from a series of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Tuple of (strength, direction) where strength is 0-1 and direction is 'up'/'down'/'flat'
    """
    try:
        if len(values) < 3:
            return 0.0, 'flat'
            
        # Linear regression to find trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and correlation
        slope, intercept = np.polyfit(x, y, 1)
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Determine direction
        if slope > 0:
            direction = 'up'
        elif slope < 0:
            direction = 'down'
        else:
            direction = 'flat'
            
        # Strength is absolute correlation
        strength = abs(correlation) if not np.isnan(correlation) else 0.0
        
        logger.debug(f"Trend calculated: strength={strength:.3f}, direction={direction}")
        return float(strength), direction
        
    except Exception as e:
        logger.error(f"Error calculating trend: {e}")
        return 0.0, 'flat' 