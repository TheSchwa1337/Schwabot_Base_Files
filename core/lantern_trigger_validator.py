from __future__ import annotations
import numpy as np
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""lantern_trigger_validator – Real validation implementation.

Validates spike/dip signals against historical Ferris Wheel & Lantern timing.
Implements real validation using historical data patterns and statistical analysis.
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional
# from core.unified_math_system import unified_math  # F811: duplicate import
from datetime import datetime, timedelta
import json
import os

__all__: list[str] = [
    "LanternTriggerValidator",
    "validate_lantern_trigger",
]


@dataclass(slots=True)
class LanternTriggerValidator:
    """Real validator with historical data analysis."""

    lookback_period: float = 3600.0  # seconds
    historical_data_path: str = "./data/lantern_historical.json"
    validation_threshold: float = 0.7
    min_confidence: float = 0.6

    def __post_init__(self):
        """Initialize validator with historical data."""
        self.historical_patterns = self._load_historical_patterns()
        self.recent_triggers = []
        self.validation_stats = {
            'total_validations': 0,
            'valid_triggers': 0,
            'invalid_triggers': 0,
            'average_confidence': 0.0
        }

    def _load_historical_patterns(self) -> Dict[str, Any]:
        """Load historical trigger patterns from file."""
        try:
            if os.path.exists(self.historical_data_path):
                with open(self.historical_data_path, 'r') as f:
                    return json.load(f)
            else:
                # Generate default historical patterns
                return self._generate_default_patterns()
        except Exception as e:
            safe_print(f"Warning: Could not load historical patterns: {e}")
            return self._generate_default_patterns()

    def _generate_default_patterns(self) -> Dict[str, Any]:
        """Generate default historical patterns for validation."""
        return {
            'ferris_wheel_patterns': {
                'cycle_duration': 3600,  # 1 hour cycles
                'spike_threshold': 0.05,  # 5% price movement
                'dip_threshold': -0.03,   # -3% price movement
                'confidence_weights': {
                    'timing': 0.3,
                    'magnitude': 0.4,
                    'volume': 0.2,
                    'volatility': 0.1
                }
            },
            'lantern_patterns': {
                'signal_duration': 300,   # 5 minute signals
                'frequency_range': [0.1, 10.0],  # Hz
                'amplitude_threshold': 0.02,
                'phase_coherence': 0.8
            },
            'market_regime_patterns': {
                'bull_market': {
                    'spike_probability': 0.7,
                    'dip_probability': 0.3,
                    'average_magnitude': 0.04
                },
                'bear_market': {
                    'spike_probability': 0.3,
                    'dip_probability': 0.7,
                    'average_magnitude': -0.04
                },
                'sideways_market': {
                    'spike_probability': 0.5,
                    'dip_probability': 0.5,
                    'average_magnitude': 0.02
                }
            }
        }

    def validate(self, trigger_packet: Dict[str, Any]) -> bool:
        """Validate trigger using historical data and statistical analysis."""
        try:
            # Extract trigger information
            trigger_type = trigger_packet.get('type', 'unknown')
            timestamp = trigger_packet.get('timestamp', datetime.now())
            price_change = trigger_packet.get('price_change', 0.0)
            volume = trigger_packet.get('volume', 0.0)
            volatility = trigger_packet.get('volatility', 0.0)

            # Calculate validation confidence
            confidence = self._calculate_validation_confidence(
                trigger_type, price_change, volume, volatility, timestamp
            )

            # Update statistics
            self._update_validation_stats(confidence)

            # Store recent trigger
            self.recent_triggers.append({
                'timestamp': timestamp,
                'type': trigger_type,
                'price_change': price_change,
                'confidence': confidence,
                'valid': confidence >= self.validation_threshold
            })

            # Keep only recent triggers
            if len(self.recent_triggers) > 1000:
                self.recent_triggers = self.recent_triggers[-500:]

            # Return validation result
            return confidence >= self.validation_threshold

        except Exception as e:
            safe_print(f"Error in trigger validation: {e}")
            return False  # Fail safe - reject if validation fails

    def _calculate_validation_confidence(
        self,
        trigger_type: str,
        price_change: float,
        volume: float,
        volatility: float,
        timestamp: datetime
    ) -> float:
        """Calculate validation confidence using multiple factors."""
        try:
            confidence_scores = []

            # 1. Timing validation (Ferris Wheel cycles)
            timing_score = self._validate_timing(timestamp)
            confidence_scores.append(timing_score * 0.3)

            # 2. Magnitude validation
            magnitude_score = self._validate_magnitude(trigger_type, price_change)
            confidence_scores.append(magnitude_score * 0.4)

            # 3. Volume validation
            volume_score = self._validate_volume(volume)
            confidence_scores.append(volume_score * 0.2)

            # 4. Volatility validation
            volatility_score = self._validate_volatility(volatility)
            confidence_scores.append(volatility_score * 0.1)

            # Calculate weighted average
            total_confidence = sum(confidence_scores)

            # Apply market regime adjustment
            market_adjustment = self._get_market_regime_adjustment(trigger_type, price_change)
            total_confidence *= market_adjustment

            return np.clip(total_confidence, 0.0, 1.0)

        except Exception as e:
            safe_print(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence

    def _validate_timing(self, timestamp: datetime) -> float:
        """Validate trigger timing against Ferris Wheel cycles."""
        try:
            # Check if timestamp aligns with known cycle patterns
            cycle_duration = self.historical_patterns['ferris_wheel_patterns']['cycle_duration']

            # Calculate time since epoch
            epoch_time = timestamp.timestamp()
            cycle_position = (epoch_time % cycle_duration) / cycle_duration

            # Check if timing is in a valid window (e.g., within 10% of cycle boundaries)
            timing_tolerance = 0.1
            if cycle_position <= timing_tolerance or cycle_position >= (1 - timing_tolerance):
                return 0.9  # High confidence for cycle-aligned triggers
            else:
                return 0.3  # Lower confidence for off-cycle triggers

        except Exception as e:
            safe_print(f"Error in timing validation: {e}")
            return 0.5

    def _validate_magnitude(self, trigger_type: str, price_change: float) -> float:
        """Validate price change magnitude."""
        try:
            patterns = self.historical_patterns['ferris_wheel_patterns']

            if trigger_type == 'spike':
                threshold = patterns['spike_threshold']
                if unified_math.abs(price_change) >= threshold:
                    return 0.9
                else:
                    return 0.3
            elif trigger_type == 'dip':
                threshold = unified_math.abs(patterns['dip_threshold'])
                if unified_math.abs(price_change) >= threshold:
                    return 0.9
                else:
                    return 0.3
            else:
                return 0.5  # Unknown trigger type

        except Exception as e:
            safe_print(f"Error in magnitude validation: {e}")
            return 0.5

    def _validate_volume(self, volume: float) -> float:
        """Validate trading volume."""
        try:
            # Normalize volume to 0-1 range (assuming typical volume range)
            normalized_volume = unified_math.min(volume / 1000000, 1.0)  # Assume 1M is max volume
            return normalized_volume

        except Exception as e:
            safe_print(f"Error in volume validation: {e}")
            return 0.5

    def _validate_volatility(self, volatility: float) -> float:
        """Validate market volatility."""
        try:
            # Higher volatility can indicate more reliable signals
            normalized_volatility = unified_math.min(volatility / 0.1, 1.0)  # Assume 10% is max volatility
            return normalized_volatility

        except Exception as e:
            safe_print(f"Error in volatility validation: {e}")
            return 0.5

    def _get_market_regime_adjustment(self, trigger_type: str, price_change: float) -> float:
        """Get market regime adjustment factor."""
        try:
            # Determine current market regime based on recent price changes
            if len(self.recent_triggers) < 10:
                return 1.0  # Default adjustment

            recent_changes = [t['price_change'] for t in self.recent_triggers[-10:]]
            avg_change = unified_math.unified_math.mean(recent_changes)

            if avg_change > 0.02:  # Bull market
                regime = 'bull_market'
            elif avg_change < -0.02:  # Bear market
                regime = 'bear_market'
            else:  # Sideways market
                regime = 'sideways_market'

            patterns = self.historical_patterns['market_regime_patterns'][regime]

            if trigger_type == 'spike':
                return patterns['spike_probability']
            elif trigger_type == 'dip':
                return patterns['dip_probability']
            else:
                return 0.5

        except Exception as e:
            safe_print(f"Error in market regime adjustment: {e}")
            return 1.0

    def _update_validation_stats(self, confidence: float) -> None:
        """Update validation statistics."""
        try:
            self.validation_stats['total_validations'] += 1
            if confidence >= self.validation_threshold:
                self.validation_stats['valid_triggers'] += 1
            else:
                self.validation_stats['invalid_triggers'] += 1

            # Update average confidence
            total = self.validation_stats['total_validations']
            current_avg = self.validation_stats['average_confidence']
            self.validation_stats['average_confidence'] = (current_avg * (total - 1) + confidence) / total

        except Exception as e:
            safe_print(f"Error updating validation stats: {e}")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()

    def save_historical_patterns(self) -> None:
        """Save current historical patterns to file."""
        try:
            os.makedirs(os.path.dirname(self.historical_data_path), exist_ok=True)
            with open(self.historical_data_path, 'w') as f:
                json.dump(self.historical_patterns, f, indent=2, default=str)
        except Exception as e:
            safe_print(f"Error saving historical patterns: {e}")


def validate_lantern_trigger(
    trigger_packet: Dict[str, Any],
) -> bool:
    """Stateless helper around :py:meth:`LanternTriggerValidator.validate`."""
    return LanternTriggerValidator().validate(trigger_packet)
