# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
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
"""
Market Signal Processor - Schwabot UROS v1.0
============================================

Implements a market signal analysis engine using Fast Fourier Transform (FFT)
to identify patterns and generate trading signals from time-series data.

Core Concepts (Analogies):
- 'Bit Phase': Represents different analysis resolutions or sensitivities.
- 'Matrix Basket': A data structure for grouping assets and their weights,
  informed by signal analysis.
- 'Quantum State' Analogy: The FFT's output (frequencies and magnitudes) is
  treated as a state vector for pattern matching and analysis.
- Hashing: FFT results are hashed to create a unique 'fingerprint' for
  fast pattern comparison and detection.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

# from core.unified_math_system import unified_math  # F811: duplicate import
from scipy.signal import get_window
import numpy as np

logger = logging.getLogger(__name__)


# --- Configuration and Data Structures ---

class AnalysisResolution(Enum):
    """Defines the resolution for waveform analysis."""
    LOW = 4
    MEDIUM = 8
    HIGH = 42


@dataclass
class SignalAnalysis:
    """Represents the results of an FFT signal analysis."""
    name: str
    frequencies: np.ndarray
    magnitudes: np.ndarray
    hash_signature: str
    resolution: AnalysisResolution
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetBasket:
    """A basket of assets with calculated weights for coordinated trading."""
    basket_id: str
    resolution: AnalysisResolution
    asset_weights: Dict[str, float]
    resonance_score: float
    hash_signature: str
    timestamp: datetime = field(default_factory=datetime.now)


# --- Core Engine ---

class DLTWaveformEngine:
    """DLT Waveform Engine - Wrapper for MarketSignalProcessor for compatibility."""

    def __init__(self, history_size: int = 1000, gpu_enabled: bool = False):
        """Initialize DLT Waveform Engine."""
        self.processor = MarketSignalProcessor(history_size, gpu_enabled)
        logger.info("DLTWaveformEngine initialized (wrapping MarketSignalProcessor)")

    def process_signal(self, *args, **kwargs):
        """Process signal using the underlying MarketSignalProcessor."""
        return self.processor.process_signal(*args, **kwargs)

    def find_similar_patterns(self, *args, **kwargs):
        """Find similar patterns using the underlying MarketSignalProcessor."""
        return self.processor.find_similar_patterns(*args, **kwargs)

    def create_asset_basket(self, *args, **kwargs):
        """Create asset basket using the underlying MarketSignalProcessor."""
        return self.processor.create_asset_basket(*args, **kwargs)

    def get_trading_signals(self, *args, **kwargs):
        """Get trading signals using the underlying MarketSignalProcessor."""
        return self.processor.get_trading_signals(*args, **kwargs)


class MarketSignalProcessor:
    """
    Analyzes market data streams using FFT to detect patterns and generate signals.

    Mathematical Foundation:
    - FFT: Computes the Discrete Fourier Transform of a signal, revealing its
      frequency components.
    - Windowing: Applies a window function (e.g., Hann) to the signal before FFT
      to reduce spectral leakage.
    - Hashing: Uses SHA-256 on the FFT magnitudes to create a reproducible
      fingerprint of the signal's pattern.
    - Pattern Matching: Compares hashes of new signals to a history of stored
      hashes to find recurring patterns.
    """

    def __init__(self, history_size: int = 1000, gpu_enabled: bool = False):
        self.history_size = history_size
        self.analysis_history: List[SignalAnalysis] = []
        self.baskets: Dict[str, AssetBasket] = {}
        self.gpu_enabled = gpu_enabled and self._check_gpu_availability()
        self.xp = self._initialize_array_library()
        logger.info(
            f"MarketSignalProcessor initialized. GPU support: {self.gpu_enabled}"
        )

    def _check_gpu_availability(self) -> bool:
        """Checks if CuPy is installed for GPU acceleration."""
        try:
            import cupy
            logger.info("CuPy found. GPU acceleration is available.")
            return True
        except ImportError:
            logger.info("CuPy not found. Using NumPy for all calculations.")
            return False

    def _initialize_array_library(self):
        """Initializes the array library (NumPy or CuPy)."""
        if self.gpu_enabled:
            import cupy
            return cupy
        return np

    def process_signal(
        self,
        name: str,
        signal_data: np.ndarray,
        sample_rate: float,
        resolution: AnalysisResolution = AnalysisResolution.MEDIUM,
        window_type: str = "hann",
    ) -> SignalAnalysis:
        """
        Processes a raw time-series signal using FFT and returns the analysis.

        Args:
            name: A unique name for this signal analysis.
            signal_data: A 1D NumPy array of time-series data.
            sample_rate: The sample rate of the signal data.
            resolution: The analysis resolution to use.
            window_type: The type of window function to apply before FFT.

        Returns:
            A SignalAnalysis object containing the results.
        """
        if signal_data.ndim != 1 or signal_data.size == 0:
            raise ValueError("signal_data must be a non-empty 1D array.")

        # --- 1. Signal Preparation ---
        signal_gpu = self.xp.asarray(signal_data)
        n_points = len(signal_gpu)
        window = self.xp.asarray(get_window(window_type, n_points))
        windowed_signal = signal_gpu * window

        # --- 2. FFT Calculation ---
        fft_result = self.xp.fft.rfft(windowed_signal)
        frequencies = self.xp.fft.rfftfreq(n_points, d=1.0 / sample_rate)
        magnitudes = self.xp.unified_math.abs(fft_result)

        # --- 3. Hashing ---
        hash_signature = self._generate_signal_hash(name, magnitudes, resolution)

        # --- 4. Store Analysis ---
        analysis = SignalAnalysis(
                name=name,
            frequencies=self.xp.asnumpy(frequencies),
            magnitudes=self.xp.asnumpy(magnitudes),
                hash_signature=hash_signature,
            resolution=resolution,
            metadata={"sample_rate": sample_rate, "window": window_type},
        )
        self._add_to_history(analysis)

        logger.info(
            f"Processed signal '{name}' ({n_points} points). "
            f"Hash: {hash_signature[:10]}..."
        )
        return analysis

    def _add_to_history(self, analysis: SignalAnalysis):
        """Adds a new analysis to the history, maintaining size limits."""
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > self.history_size:
            self.analysis_history.pop(0)

    def _generate_signal_hash(
        self, name: str, magnitudes: np.ndarray, resolution: AnalysisResolution
    ) -> str:
        """Creates a SHA-256 hash from the signal's dominant frequencies."""
        # Normalize and discretize magnitudes to create a stable hash
        normalized_magnitudes = magnitudes / (self.xp.unified_math.max(magnitudes) + 1e-9)
        # Use a number of bins related to the resolution
        bins = resolution.value * 10
        quantized_magnitudes = (normalized_magnitudes * bins).astype(self.xp.int32)

        hasher = hashlib.sha256()
        hasher.update(name.encode())
        hasher.update(quantized_magnitudes.tobytes())
        hasher.update(resolution.name.encode())
        return hasher.hexdigest()

    def find_similar_patterns(
        self, new_analysis: SignalAnalysis, similarity_threshold: float = 0.90
    ) -> List[SignalAnalysis]:
        """
        Finds historical analyses with similar waveform patterns.
        Similarity is determined by comparing the Jaccard similarity of hashes.
        """
        similar_analyses = []
        new_hash_set = set(new_analysis.hash_signature)

        for old_analysis in self.analysis_history:
            if old_analysis.hash_signature == new_analysis.hash_signature:
                continue  # Skip self

            old_hash_set = set(old_analysis.hash_signature)

            intersection = len(new_hash_set.intersection(old_hash_set))
            union = len(new_hash_set.union(old_hash_set))

            jaccard_similarity = intersection / union if union > 0 else 0

            if jaccard_similarity >= similarity_threshold:
                similar_analyses.append(old_analysis)

        return similar_analyses

    def create_asset_basket(
        self,
        market_data: Dict[str, float],
        resolution: AnalysisResolution = AnalysisResolution.MEDIUM,
    ) -> AssetBasket:
        """
        Creates an asset basket based on current market data and a signal profile.
        This is a placeholder for a more complex strategy.
        """
        basket_id = f"basket_{int(time.time())}"

        # Example weighting strategy: inverse volatility
        total_inverse_vol = sum(1.0 / (v + 1e-9) for v in market_data.values())
        asset_weights = {
            asset: (1.0 / (vol + 1e-9)) / total_inverse_vol
            for asset, vol in market_data.items()
        }

        # Resonance score can be a measure of how well the basket fits a model
        resonance_score = unified_math.unified_math.std(list(asset_weights.values()))

        hasher = hashlib.sha256()
        hasher.update(json.dumps(asset_weights, sort_keys=True).encode())
        hash_signature = hasher.hexdigest()

        basket = AssetBasket(
            basket_id=basket_id,
            resolution=resolution,
            asset_weights=asset_weights,
            resonance_score=resonance_score,
            hash_signature=hash_signature,
        )
        self.baskets[basket_id] = basket
        logger.info(f"Created asset basket '{basket_id}'")
        return basket

    def get_trading_signals(self, latest_analysis: SignalAnalysis) -> List[Dict]:
        """
        Generates trading signals based on pattern detection.
        This is a simplified example strategy.
        """
        signals = []
        similar = self.find_similar_patterns(latest_analysis, similarity_threshold=0.92)

        if len(similar) > 1:
            # If a strong pattern is detected (multiple similar past events)
            dominant_freq = latest_analysis.frequencies[
                np.argmax(latest_analysis.magnitudes)
            ]
            signal = {
                "type": "PATTERN_CONFIRMED",
                "strength": len(similar),
                "dominant_frequency": float(dominant_freq),
                "timestamp": datetime.now(),
                "recommendation": "MONITOR_FOR_ENTRY",
                "reason": f"{len(similar)} similar past patterns found.",
            }
            signals.append(signal)

        # Signal based on high frequency energy
        high_freq_threshold = (unified_math.unified_math.max(latest_analysis.frequencies)) * 0.75
        high_freq_magnitudes = latest_analysis.magnitudes[
            latest_analysis.frequencies > high_freq_threshold
        ]
        total_magnitude = np.sum(latest_analysis.magnitudes)
        high_freq_energy_ratio = np.sum(high_freq_magnitudes) / total_magnitude

        if high_freq_energy_ratio > 0.4:  # 40% of energy is in high frequencies
            signals.append(
                {
                    "type": "HIGH_FREQUENCY_ALERT",
                    "energy_ratio": float(high_freq_energy_ratio),
                    "timestamp": datetime.now(),
                    "recommendation": "CAUTION_VOLATILITY",
                    "reason": "High energy in upper frequency bands suggests instability.",
                }
            )

        return signals


def main():
    """Main function to demonstrate MarketSignalProcessor usage."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Setup ---
    engine = MarketSignalProcessor(gpu_enabled=False)
    sample_rate = 44100
    n_points = sample_rate * 2  # 2 seconds of data

    # --- Generate a base signal with some characteristic frequencies ---
    time_vector = np.linspace(0, 2, n_points, endpoint=False)
    signal1 = (
        np.unified_math.sin(2 * np.pi * 150 * time_vector) +  # 150 Hz component
        0.5 * np.unified_math.sin(2 * np.pi * 400 * time_vector) +  # 400 Hz component
        np.random.normal(0, 0.2, n_points)  # Some noise
    )
    safe_print("Processing first signal...")
    analysis1 = engine.process_signal(
        "Signal_A", signal1, sample_rate, resolution=AnalysisResolution.MEDIUM
    )

    # --- Generate a second, similar signal ---
    time.sleep(1)
    signal2 = (
        np.unified_math.sin(2 * np.pi * 150 * time_vector + 0.2) +  # Same freq, phase shifted
        0.5 * np.unified_math.sin(2 * np.pi * 400 * time_vector) +
        np.random.normal(0, 0.2, n_points)
    )
    safe_print("\nProcessing second (similar) signal...")
    analysis2 = engine.process_signal(
        "Signal_B", signal2, sample_rate, resolution=AnalysisResolution.MEDIUM
    )

    # --- Find patterns and generate signals ---
    safe_print(f"\nFinding patterns similar to '{analysis2.name}'...")
    similar_patterns = engine.find_similar_patterns(analysis2)
    if similar_patterns:
        safe_print(f"Found {len(similar_patterns)} similar pattern(s).")
        for p in similar_patterns:
            safe_print(f"  - Similar analysis: '{p.name}' from {p.timestamp}")
    else:
        safe_print("No highly similar patterns found.")

    safe_print(f"\nGenerating trading signals for '{analysis2.name}'...")
    trading_signals = engine.get_trading_signals(analysis2)
    if trading_signals:
        print(json.dumps(trading_signals, indent=2))
    else:
        safe_print("No specific trading signals generated.")

    # --- Demonstrate asset basket creation ---
    safe_print("\nCreating an asset basket...")
    mock_market_volatility = {"BTC": 0.02, "ETH": 0.035, "SOL": 0.05}
    basket = engine.create_asset_basket(
        mock_market_volatility, resolution=AnalysisResolution.MEDIUM
    )
    safe_print(f"Basket '{basket.basket_id}' created with weights:")
    print(json.dumps(basket.asset_weights, indent=2))


if __name__ == "__main__":
    main()
