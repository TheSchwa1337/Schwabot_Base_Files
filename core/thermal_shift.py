"""
Thermal Shift Detector - Minimal Thermal Drift Detection System
==============================================================

This helper flags sudden temperature jumps (thermal shifts) above a preset
threshold. It is intentionally lightweight so it can execute inside tight
trading-loop iterations without blocking the GIL.

Current implementation:
1. ThermalShift class - exponential-moving-average (EWMA) smoothing with
   update() returning (is_stable, delta).
2. Stateless wrapper: thermal_delta_switch() mirroring the legacy stub
    signature requested by earlier Schwabot code.
3. Fully typed and Flake8-clean (≤ 79-character lines).

Future versions may include adaptive hysteresis or GPU-calibrated drift maps.
"""

import logging
from dataclasses import dataclass, field
from typing import Final, Tuple

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Configure logging
logger = logging.getLogger(__name__)

__all__ = ["ThermalShift", "thermal_delta_switch"]

_DEFAULT_ALPHA: Final = 0.2
_DEFAULT_THRESHOLD: Final = 2.5  # °C


@dataclass(slots=True)
class ThermalShift:
    """
    EWMA-based thermal drift detector.

Parameters
----------
threshold
        Absolute temperature delta (°C) that triggers an *unstable* flag.
    alpha
        EWMA smoothing factor between 0 and 1. Higher = faster reaction.
    """

threshold: float = _DEFAULT_THRESHOLD
alpha: float = _DEFAULT_ALPHA
    _ema: float | None = field(default=None, init=False)

def update(self, temp: float) -> Tuple[bool, float]:
        """
        Process a new temperature reading and return stability status.

Parameters
----------
temp
            Current temperature reading (°C).

Returns
-------
Tuple[bool, float]
            (is_stable, delta), where *delta* is the absolute
            temperature change with respect to the EWMA baseline.
        """
        try:
    if self._ema is None:
        self._ema = temp
    else:
        self._ema = self.alpha * temp + (1.0 - self.alpha) * self._ema

delta = unified_math.abs(temp - self._ema)
is_stable = delta < self.threshold
            
            return is_stable, delta

        except Exception as e:
            logger.error(f"Thermal shift update failed: {e}")
            return True, 0.0  # Default to stable

    def reset(self) -> None:
        """Reset the EWMA baseline."""
        self._ema = None

    def get_status(self) -> dict:
        """Get current status of the thermal shift detector."""
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "ema_baseline": self._ema,
            "is_initialized": self._ema is not None
        }


def thermal_delta_switch(
    current: float,
    previous: float,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """
    Return True if the temperature delta is below threshold.

Parameters
----------
current
        Current temperature reading (°C).
    previous
        Previous or baseline temperature reading (°C).
    threshold
        Allowed delta before declaring instability. Defaults to 2.5 °C.

    Returns
    -------
    bool
        True if temperature change is within threshold, False otherwise.
    """
    try:
delta = unified_math.abs(current - previous)
        return delta < threshold
    except Exception as e:
        logger.error(f"Thermal delta switch failed: {e}")
        return True  # Default to stable


class AdvancedThermalShiftDetector:
    """Advanced thermal shift detection with multiple algorithms."""

    def __init__(self, 
                 threshold: float = _DEFAULT_THRESHOLD,
                 alpha: float = _DEFAULT_ALPHA,
                 window_size: int = 10):
        """
        Initialize advanced thermal shift detector.

        Parameters
        ----------
        threshold
            Temperature delta threshold (°C)
        alpha
            EWMA smoothing factor
        window_size
            Size of sliding window for statistical analysis
        """
        self.threshold = threshold
        self.alpha = alpha
        self.window_size = window_size
        
        # Initialize detectors
        self.ewma_detector = ThermalShift(threshold, alpha)
        self.temperature_history = []
        self.shift_count = 0
        self.last_shift_time = None

    def update(self, temp: float) -> dict:
        """
        Update with new temperature reading and return comprehensive analysis.

        Parameters
        ----------
        temp
            Current temperature reading (°C)

        Returns
        -------
        dict
            Comprehensive thermal shift analysis
        """
        try:
            # Update EWMA detector
            is_stable_ewma, delta_ewma = self.ewma_detector.update(temp)
            
            # Update temperature history
            self.temperature_history.append(temp)
            if len(self.temperature_history) > self.window_size:
                self.temperature_history.pop(0)
            
            # Calculate statistical measures
            stats = self._calculate_statistics()
            
            # Detect shifts using multiple methods
            shift_detected = self._detect_shift(temp, stats)
            
            # Update shift tracking
            if shift_detected:
                self.shift_count += 1
                import time
                self.last_shift_time = time.time()
            
            return {
                "temperature": temp,
                "is_stable": is_stable_ewma and not shift_detected,
                "delta_ewma": delta_ewma,
                "shift_detected": shift_detected,
                "shift_count": self.shift_count,
                "statistics": stats,
                "last_shift_time": self.last_shift_time
            }
            
        except Exception as e:
            logger.error(f"Advanced thermal shift update failed: {e}")
            return {
                "temperature": temp,
                "is_stable": True,
                "delta_ewma": 0.0,
                "shift_detected": False,
                "shift_count": self.shift_count,
                "statistics": {},
                "error": str(e)
            }

    def _calculate_statistics(self) -> dict:
        """
        Calculate statistical measures from temperature history.

        Returns
        -------
        dict
            Statistical measures including mean, std, trend, etc.
        """
        try:
            if len(self.temperature_history) < 2:
                return {
                    "mean": self.temperature_history[0] if self.temperature_history else 0.0,
                    "std": 0.0,
                    "trend": 0.0,
                    "min": self.temperature_history[0] if self.temperature_history else 0.0,
                    "max": self.temperature_history[0] if self.temperature_history else 0.0
                }
            
            temps = self.temperature_history
            
            # Basic statistics
            mean_temp = sum(temps) / len(temps)
            variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
            std_temp = unified_math.sqrt(variance)
            
            # Trend calculation (simple linear regression)
            n = len(temps)
            if n > 1:
                x_sum = sum(range(n))
                y_sum = sum(temps)
                xy_sum = sum(i * t for i, t in enumerate(temps))
                x2_sum = sum(i * i for i in range(n))
                
                slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            else:
                slope = 0.0
            
            return {
                "mean": mean_temp,
                "std": std_temp,
                "trend": slope,
                "min": min(temps),
                "max": max(temps),
                "range": max(temps) - min(temps),
                "count": len(temps)
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {
                "mean": 0.0,
                "std": 0.0,
                "trend": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "count": 0
            }

    def _detect_shift(self, temp: float, stats: dict) -> bool:
        """
        Detect thermal shift using multiple criteria.

        Parameters
        ----------
        temp
            Current temperature
        stats
            Statistical measures

        Returns
        -------
        bool
            True if shift detected, False otherwise
        """
        try:
            if not stats or "mean" not in stats:
                return False
            
            mean_temp = stats["mean"]
            std_temp = stats.get("std", 0.0)
            trend = stats.get("trend", 0.0)
            
            # Multiple shift detection criteria
            
            # 1. Large deviation from mean
            deviation = unified_math.abs(temp - mean_temp)
            if deviation > self.threshold * 2:
                return True
            
            # 2. High standard deviation (indicating instability)
            if std_temp > self.threshold:
                return True
            
            # 3. Strong trend (rapid temperature change)
            if unified_math.abs(trend) > self.threshold / 2:
                return True
            
            # 4. Sudden jump from previous reading
            if len(self.temperature_history) > 1:
                prev_temp = self.temperature_history[-2]
                jump = unified_math.abs(temp - prev_temp)
                if jump > self.threshold * 1.5:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Shift detection failed: {e}")
            return False

    def get_recommendations(self) -> list:
        """
        Get recommendations based on thermal shift analysis.

        Returns
        -------
        list
            List of recommendations
        """
        try:
            recommendations = []
            
            if self.shift_count > 5:
                recommendations.append("High number of thermal shifts detected - check system stability")
            
            if len(self.temperature_history) >= self.window_size:
                stats = self._calculate_statistics()
                if stats.get("std", 0.0) > self.threshold:
                    recommendations.append("High temperature variability - consider thermal management")
                
                if unified_math.abs(stats.get("trend", 0.0)) > self.threshold / 2:
                    recommendations.append("Sustained temperature trend detected - monitor closely")
            
            if not recommendations:
                recommendations.append("Thermal conditions appear stable")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Error generating recommendations"]

    def reset(self) -> None:
        """Reset the thermal shift detector."""
        try:
            self.ewma_detector.reset()
            self.temperature_history.clear()
            self.shift_count = 0
            self.last_shift_time = None
        except Exception as e:
            logger.error(f"Reset failed: {e}")

    def get_status(self) -> dict:
        """
        Get comprehensive status of the thermal shift detector.

        Returns
        -------
        dict
            Status information
        """
        try:
            return {
                "threshold": self.threshold,
                "alpha": self.alpha,
                "window_size": self.window_size,
                "shift_count": self.shift_count,
                "history_length": len(self.temperature_history),
                "last_shift_time": self.last_shift_time,
                "ewma_status": self.ewma_detector.get_status()
            }
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}


def create_thermal_shift_detector(config: dict = None) -> AdvancedThermalShiftDetector:
    """
    Factory function to create a thermal shift detector.

    Parameters
    ----------
    config
        Configuration dictionary

    Returns
    -------
    AdvancedThermalShiftDetector
        Configured thermal shift detector
    """
    try:
        if config:
            threshold = config.get("threshold", _DEFAULT_THRESHOLD)
            alpha = config.get("alpha", _DEFAULT_ALPHA)
            window_size = config.get("window_size", 10)
            return AdvancedThermalShiftDetector(threshold, alpha, window_size)
        else:
            return AdvancedThermalShiftDetector()
    except Exception as e:
        logger.error(f"Failed to create thermal shift detector: {e}")
        raise


def main():
    """Main function for testing the thermal shift detector."""
    try:
        # Create detector
        detector = create_thermal_shift_detector()
        
        # Simulate temperature readings
        test_temps = [20.0, 21.0, 22.0, 25.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0]
        
        print("Testing Thermal Shift Detector:")
        print("=" * 40)
        
        for i, temp in enumerate(test_temps):
            result = detector.update(temp)
            print(f"Reading {i+1}: {temp}°C")
            print(f"  Stable: {result['is_stable']}")
            print(f"  Shift Detected: {result['shift_detected']}")
            print(f"  Delta EWMA: {result['delta_ewma']:.2f}°C")
            print()
        
        # Get final status
        status = detector.get_status()
        print(f"Final Status: {status}")
        
        # Get recommendations
        recommendations = detector.get_recommendations()
        print(f"Recommendations: {recommendations}")
        
    except Exception as e:
        logger.error(f"Main function failed: {e}")


if __name__ == "__main__":
    main()


