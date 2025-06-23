#!/usr/bin/env python3
"""
dlt_waveform_engine.py - Discrete Log-Time Waveform Engine for Schwabot.

Builds discrete logic-based waveform profiles to monitor volatility and detect
momentum-based changes across time ticks. Serves as a reactive core for
timing entries and exits.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any

from core.utils.math_utils import (
    calculate_tick_acceleration,
    waveform_pattern_match,
    moving_average,
)

logger = logging.getLogger(__name__)


class DLTWaveformEngine:
    """
    Analyzes price and time data to identify momentum shifts and patterns.
    """

    def __init__(self, history_size: int = 200, pattern_threshold: float = 0.85):
        """
        Initialize the DLT Waveform Engine.

        Args:
            history_size: The number of recent data points to keep for analysis.
            pattern_threshold: The confidence score required to confirm a pattern match.
        """
        self.history_size = history_size
        self.pattern_threshold = pattern_threshold
        
        self.price_history: List[float] = []
        self.time_history: List[float] = []
        self.velocity_history: List[float] = []
        self.acceleration_history: List[float] = []
        
        # Store a library of known reference waveforms to match against
        self.pattern_library: Dict[str, np.ndarray] = {}

        logger.info("DLT Waveform Engine initialized.")

    def update_tick_data(self, price: float, timestamp: float) -> None:
        """

        Update the engine with a new price tick and timestamp.
        This method is the primary entry point for new data.
        """
        self.price_history.append(price)
        self.time_history.append(timestamp)

        # Trim history to maintain size
        if len(self.price_history) > self.history_size:
            self.price_history.pop(0)
            self.time_history.pop(0)
            self.velocity_history.pop(0)
            self.acceleration_history.pop(0)

        self._calculate_derivatives()

    def _calculate_derivatives(self) -> None:
        """Calculate the velocity and acceleration of price changes."""
        if len(self.price_history) < 2:
            self.velocity_history.append(0)
            self.acceleration_history.append(0)
            return

        # Calculate time and price deltas
        delta_times = np.diff(self.time_history)
        price_deltas = np.diff(self.price_history)

        # Avoid division by zero for time
        valid_mask = delta_times > 1e-10
        velocities = np.zeros_like(delta_times)
        velocities[valid_mask] = price_deltas[valid_mask] / delta_times[valid_mask]
        
        # Calculate acceleration
        accelerations = calculate_tick_acceleration(velocities, delta_times)

        # Append the latest calculated values
        self.velocity_history.append(velocities[-1] if velocities.size > 0 else 0)
        self.acceleration_history.append(accelerations[-1] if accelerations is not None and accelerations.size > 0 else 0)
        
    def add_reference_pattern(self, name: str, pattern_data: List[float]) -> None:
        """Add a new named waveform pattern to the library."""
        if not pattern_data:
            logger.warning(f"Attempted to add empty pattern '{name}'.")
            return
        self.pattern_library[name] = np.array(pattern_data)
        logger.info(f"Added reference pattern '{name}' to library.")

    def analyze_current_waveform(self) -> Dict[str, Any]:
        """
        Analyze the current waveform for pattern matches and momentum signals.

        Returns:
            A dictionary containing analysis results, including any matched patterns
            and current momentum indicators.
        """
        if len(self.price_history) < 10:
            return {"status": "insufficient_data"}

        live_waveform = np.array(self.price_history[-50:]) # Use last 50 points for matching
        
        matched_patterns = {}
        for name, reference_wave in self.pattern_library.items():
            is_match, confidence = waveform_pattern_match(
                live_wave, reference_wave, self.pattern_threshold
            )
            if is_match:
                matched_patterns[name] = confidence
                logger.debug(f"Pattern '{name}' matched with confidence {confidence:.2f}")

        # Smooth acceleration to get a clearer signal
        smooth_acceleration = moving_average(np.array(self.acceleration_history), window=5)
        
        analysis = {
            "status": "analysis_complete",
            "matched_patterns": matched_patterns,
            "current_price": self.price_history[-1],
            "current_velocity": self.velocity_history[-1],
            "current_acceleration": self.acceleration_history[-1],
            "smoothed_acceleration": smooth_acceleration[-1] if smooth_acceleration.size > 0 else 0,
            "is_accelerating": self.acceleration_history[-1] > 0 if self.acceleration_history else False,
        }
        
        # --- HOOKS INTO OTHER MODULES (Example) ---
        # if analysis['is_accelerating'] and matched_patterns:
        #     # Hooks into entry_exit_vector_analyzer.py
        #     self.signal_trade_entry(analysis)
        #
        # # Hooks into mathlib_v2 (via math_utils) and fractal_core.py
        # self.validate_with_fractal_loopback(live_waveform)

        return analysis
