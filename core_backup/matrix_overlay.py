# core/matrix_overlay.py

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MatrixOverlay:
    """"""
    Detects long-term drift and resonance patterns via FFT of alignment/entropy vectors.
    Mathematical Form: s(t) = vec_a(t) + vec_gamma(t)
    FFT: hat_s(f) = sum_{t=0}^{N-1} s(t) e^(-2pii f t / N)
    """"""

    def __init__()
        self, history_size: int = 256, low_frequency_threshold: float = 0.5, confidence_threshold: float = 0.7
    ):
        """"""
        Initializes the MatrixOverlay.

        Args:
            history_size (int): The number of data points to keep for FFT analysis.
                                 Should be a power of 2 for optimal FFT performance.
            low_frequency_threshold (float): Frequencies below this are considered long-term patterns.
            confidence_threshold (float): Amplitude confidence threshold to classify a pattern as significant.
        """"""
        self.history_size = history_size
        self.low_frequency_threshold = low_frequency_threshold
        self.confidence_threshold = confidence_threshold
        self.alignment_history: List[float] = []
        self.drift_history: List[float] = []
        logger.info(f"MatrixOverlay initialized with history size {history_size}.")

    def add_data(self, alignment_score: float, drift_resonance: float):
        """"""
        Adds new alignment and drift data points to the history.

        Args:
            alignment_score (float): The current alignment score.
            drift_resonance (float): The current drift resonance.
        """"""
        if len(self.alignment_history) >= self.history_size:
            self.alignment_history.pop(0)
            self.drift_history.pop(0)
        self.alignment_history.append(alignment_score)
        self.drift_history.append(drift_resonance)
        logger.debug(f"Added data. History size: {len(self.alignment_history)}")

    def _perform_fft(self, signal: List[float]) -> np.ndarray:
        """"""
        Performs Discrete Fourier Transform using NumPy's FFT.'
        """"""
        if not signal:
            return np.array([])
        return np.fft.fft(signal)

    def analyze_patterns(self) -> Dict[str, Any]:
        """"""
        Analyzes the historical alignment and drift data for harmonic patterns using FFT.
        Classifies pattern type (short/med/long) and links to GAN anomaly scores (conceptually).

        Returns:
            Dict[str, Any]: A dictionary containing detected harmonic patterns and analysis results.
        """"""
        if len(self.alignment_history) < self.history_size:
            logger.warning("Insufficient data for full FFT analysis. Padding with zeros.")
            # Pad with zeros to meet history_size requirements for FFT
            padded_alignment = self.alignment_history + [0.0] * (self.history_size - len(self.alignment_history))
            padded_drift = self.drift_history + [0.0] * (self.history_size - len(self.drift_history))
            current_signal_length = len(self.alignment_history)  # Actual data points
        else:
            padded_alignment = self.alignment_history
            padded_drift = self.drift_history
            current_signal_length = self.history_size

        # Construct the composite signal s(t) = alignment(t) + drift(t)
        composite_signal = np.array(padded_alignment) + np.array(padded_drift)

        if composite_signal.size == 0:
            logger.warning("Composite signal is empty. Cannot perform FFT.")
            return {"status": "no_data", "patterns_detected": False, "dominant_pattern": None}

        fft_results = self._perform_fft(list(composite_signal))
        N = len(fft_results)

        if N == 0:
            return {"status": "no_data", "patterns_detected": False, "dominant_pattern": None}

        # Calculate frequencies corresponding to FFT results
        # Frequencies are usually up to N/2 + 1 (Nyquist frequency)
        frequencies = np.fft.fftfreq(N, d=1)  # d=1 means sampling interval of 1 (tick)

        # We are interested in positive frequencies and excluding the DC component (index 0)
        positive_freq_indices = np.where(frequencies > 0.0)[0]

        if not positive_freq_indices.size > 0:
            logger.info("No positive frequencies found for analysis.")
            return {"status": "no_patterns", "patterns_detected": False, "dominant_pattern": None}

        magnitudes = np.abs(fft_results[positive_freq_indices])
        phases = np.angle(fft_results[positive_freq_indices])
        actual_frequencies = frequencies[positive_freq_indices]

        # Find dominant pattern
        if magnitudes.size == 0:
            return {"status": "no_patterns", "patterns_detected": False, "dominant_pattern": None}

        max_amplitude_idx = np.argmax(magnitudes)
        dominant_frequency = actual_frequencies[max_amplitude_idx]
        dominant_amplitude = magnitudes[max_amplitude_idx]
        dominant_phase = phases[max_amplitude_idx]

        # Calculate pattern confidence
        max_overall_amplitude = np.max(np.abs(fft_results[1:])) if N > 1 else 1.0  # Exclude DC
        if max_overall_amplitude == 0:
            pattern_confidence = 0.0
        else:
            pattern_confidence = dominant_amplitude / max_overall_amplitude

        # Classify pattern type
        pattern_type = "UNKNOWN"
        if pattern_confidence > self.confidence_threshold:
            if dominant_frequency < self.low_frequency_threshold:
                pattern_type = "LONG_TERM_DRIFT"
            elif dominant_frequency < (1.0 / (current_signal_length / 4)):  # Example for medium term
                pattern_type = "MEDIUM_TERM_RESONANCE"
            else:
                pattern_type = "SHORT_TERM_NOISE"
        else:
            pattern_type = "INSIGNIFICANT_PATTERN"

        logger.info()
            f"Pattern Analysis: Type={pattern_type}, Freq={dominant_frequency:.4f}, Ampl={dominant_amplitude:.4f}, Conf={pattern_confidence:.4f}"
        )

        return {}
            "status": "success",
                "patterns_detected": pattern_confidence > self.confidence_threshold,
                    "dominant_pattern": {}
                "frequency": dominant_frequency,
                    "amplitude": dominant_amplitude,
                        "phase": dominant_phase,
                        "confidence": pattern_confidence,
                        "type": pattern_type,
                        },
                        "all_harmonic_patterns": [  # Store top few patterns]
                {}
                    "frequency": freq,
                        "amplitude": amp,
                            "phase": pha,
                            "confidence": amp / max_overall_amplitude,
                            "type": "",
}
                for freq, amp, pha in sorted()
                    zip(actual_frequencies, magnitudes, phases), key=lambda x: x[1], reverse=True
                )[]
                    :5
                ]  # Top 5
            ],
}
    def get_matrix_overlay_state(self) -> Dict[str, Any]:
        """"""
        Returns the current state of the matrix overlay, including historical data.
        """"""
        return {}
            "alignment_history_size": len(self.alignment_history),
                "drift_history_size": len(self.drift_history),
                    "current_history_length": len(self.alignment_history),
                    "config": {}
                "history_size": self.history_size,
                    "low_frequency_threshold": self.low_frequency_threshold,
                        "confidence_threshold": self.confidence_threshold,
                        },
}
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example Usage
    matrix_overlay = MatrixOverlay(history_size=64, low_frequency_threshold=0.8, confidence_threshold=0.5)

    print("\n--- Simulating Data for Matrix Overlay ---")

    # Simulate data with a long-term drift and some resonance
    t_vals = np.linspace(0, 10 * np.pi, 200)  # Simulate 200 ticks
    for i in range(200):
        alignment = 0.5 + 0.2 * np.sin(t_vals[i] / (2 * np.pi * 5)) + np.random.normal(0, 0.5)  # Slow sine + noise
        drift = 0.1 + 0.5 * np.cos(t_vals[i] / (2 * np.pi * 2)) + np.random.normal(0, 0.2)  # Faster sine + noise
        matrix_overlay.add_data(alignment, drift)

        if (i + 1) % 10 == 0:  # Analyze patterns periodically
            print(f"\nAnalysis at tick {i+1}:")
            results = matrix_overlay.analyze_patterns()
            print(f"  Patterns Detected: {results['patterns_detected']}")
            if results["dominant_pattern"]:
                dp = results["dominant_pattern"]
                print()
                    f"  Dominant Pattern Type: {dp['type']} (Freq: {dp['frequency']:.4f}, Ampl: {dp['amplitude']:.4f}, Conf: {dp['confidence']:.4f})"
                )
            else:
                print("  No dominant pattern identified.")

    print("\n--- Testing with short history ---")
    short_overlay = MatrixOverlay(history_size=16)
    short_overlay.add_data(0.1, 0.2)
    short_overlay.add_data(0.15, 0.25)
    short_results = short_overlay.analyze_patterns()
    print(f"Short history analysis: {short_results['status']}")

    print("\n--- Current Matrix Overlay State ---")
    print(matrix_overlay.get_matrix_overlay_state())
