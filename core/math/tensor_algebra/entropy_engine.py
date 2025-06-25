from __future__ import annotations

#!/usr/bin/env python3
"""
Entropy Engine - Advanced Entropy-Based Signal Processing
========================================================

Provides advanced entropy-based signal processing, wave detection,
and pattern analysis for the Schwabot trading system.

Core Functions:
- entropy_filter: Apply entropy-based filtering to signals
- calculate_dynamic_entropy: Calculate dynamic entropy over time
- entropy_wave_detection: Detect entropy waves and patterns
- entropy_pattern_analysis: Analyze entropy patterns
- entropy_based_clustering: Perform entropy-based clustering
"""


import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Union, Dict, Any
import logging
from scipy.stats import entropy
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class EntropyEngine:
    """
    Advanced Entropy Engine for Schwabot Trading System.

    This engine provides comprehensive entropy-based signal processing
    and pattern analysis capabilities.
    """

    def __init__(self):
        """Initialize the entropy engine."""
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.min_entropy_threshold = 0.1  # Minimum entropy threshold
        self.max_entropy_threshold = 0.9  # Maximum entropy threshold
        self.wave_detection_sensitivity = 0.5  # Wave detection sensitivity

        logger.info("Entropy Engine initialized")

    def entropy_filter(self, signal: NDArray, threshold: float = 0.5) -> NDArray:
        """
        Apply entropy-based filtering to signal.

        Args:
            signal: Input signal array
            threshold: Entropy threshold for filtering

        Returns:
            Filtered signal array
        """
        try:
            if len(signal) == 0:
                return signal.copy()

            # Calculate local entropy for each point
            filtered_signal = np.zeros_like(signal)
            window_size = min(10, len(signal) // 4)

            for i in range(len(signal)):
                # Define local window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(signal), i + window_size // 2 + 1)
                local_window = signal[start_idx:end_idx]

                # Calculate local entropy
                local_entropy = self._calculate_local_entropy(local_window)

                # Apply filtering based on entropy threshold
                if local_entropy > threshold:
                    filtered_signal[i] = signal[i]
                else:
                    # Apply smoothing for low entropy regions
                    filtered_signal[i] = np.mean(local_window)

            return filtered_signal

        except Exception as e:
            logger.error(f"Entropy filtering failed: {e}")
            return signal.copy()

    def calculate_dynamic_entropy(self, signal: NDArray, window: int = 20) -> float:
        """
        Calculate dynamic entropy over a sliding window.

        Args:
            signal: Input signal array
            window: Window size for entropy calculation

        Returns:
            Dynamic entropy value
        """
        try:
            if len(signal) < window:
                return self._calculate_local_entropy(signal)

            # Calculate entropy for the most recent window
            recent_window = signal[-window:]
            return self._calculate_local_entropy(recent_window)

        except Exception as e:
            logger.error(f"Dynamic entropy calculation failed: {e}")
            return 0.5

    def entropy_wave_detection(self, signal: NDArray,
                             min_peak_distance: int = 5) -> Dict[str, Any]:
        """
        Detect entropy waves and patterns in signal.

        Args:
            signal: Input signal array
            min_peak_distance: Minimum distance between peaks

        Returns:
            Dictionary containing wave detection results
        """
        try:
            if len(signal) < 10:
                return {
                    'peaks': [],
                    'troughs': [],
                    'wave_count': 0,
                    'wave_frequency': 0.0,
                    'wave_amplitude': 0.0
                }

            # Calculate entropy series
            entropy_series = []
            window_size = min(10, len(signal) // 4)

            for i in range(len(signal)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(signal), i + window_size // 2 + 1)
                local_window = signal[start_idx:end_idx]
                local_entropy = self._calculate_local_entropy(local_window)
                entropy_series.append(local_entropy)

            entropy_series = np.array(entropy_series)

            # Detect peaks and troughs
            peaks, _ = find_peaks(entropy_series, distance=min_peak_distance)
            troughs, _ = find_peaks(-entropy_series, distance=min_peak_distance)

            # Calculate wave statistics
            wave_count = len(peaks)
            wave_frequency = wave_count / len(signal) if len(signal) > 0 else 0.0

            # Calculate average wave amplitude
            if len(peaks) > 0:
                peak_values = entropy_series[peaks]
                wave_amplitude = float(np.mean(peak_values))
            else:
                wave_amplitude = 0.0

            return {
                'peaks': peaks.tolist(),
                'troughs': troughs.tolist(),
                'wave_count': wave_count,
                'wave_frequency': wave_frequency,
                'wave_amplitude': wave_amplitude,
                'entropy_series': entropy_series.tolist()
            }

        except Exception as e:
            logger.error(f"Entropy wave detection failed: {e}")
            return {
                'peaks': [],
                'troughs': [],
                'wave_count': 0,
                'wave_frequency': 0.0,
                'wave_amplitude': 0.0,
                'entropy_series': []
            }

    def entropy_pattern_analysis(self, signal: NDArray,
                               pattern_length: int = 10) -> Dict[str, Any]:
        """
        Analyze entropy patterns in signal.

        Args:
            signal: Input signal array
            pattern_length: Length of patterns to analyze

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if len(signal) < pattern_length:
                return {
                    'pattern_types': [],
                    'pattern_frequencies': {},
                    'dominant_pattern': None,
                    'pattern_stability': 0.0
                }

            # Extract patterns
            patterns = []
            for i in range(len(signal) - pattern_length + 1):
                pattern = signal[i:i + pattern_length]
                patterns.append(pattern)

            if not patterns:
                return {
                    'pattern_types': [],
                    'pattern_frequencies': {},
                    'dominant_pattern': None,
                    'pattern_stability': 0.0
                }

            # Calculate entropy for each pattern
            pattern_entropies = []
            for pattern in patterns:
                pattern_entropy = self._calculate_local_entropy(pattern)
                pattern_entropies.append(pattern_entropy)

            pattern_entropies = np.array(pattern_entropies)

            # Classify patterns based on entropy
            pattern_types = []
            for entropy_val in pattern_entropies:
                if entropy_val < 0.3:
                    pattern_types.append('low_entropy')
                elif entropy_val < 0.7:
                    pattern_types.append('medium_entropy')
                else:
                    pattern_types.append('high_entropy')

            # Calculate pattern frequencies
            pattern_frequencies = {}
            for pattern_type in set(pattern_types):
                pattern_frequencies[pattern_type] = pattern_types.count(pattern_type) / len(pattern_types)

            # Find dominant pattern
            dominant_pattern = max(pattern_frequencies, key=pattern_frequencies.get) if pattern_frequencies else None

            # Calculate pattern stability (inverse of entropy variance)
            pattern_stability = 1.0 / (1.0 + np.var(pattern_entropies))

            return {
                'pattern_types': pattern_types,
                'pattern_frequencies': pattern_frequencies,
                'dominant_pattern': dominant_pattern,
                'pattern_stability': float(pattern_stability),
                'pattern_entropies': pattern_entropies.tolist()
            }

        except Exception as e:
            logger.error(f"Entropy pattern analysis failed: {e}")
            return {
                'pattern_types': [],
                'pattern_frequencies': {},
                'dominant_pattern': None,
                'pattern_stability': 0.0,
                'pattern_entropies': []
            }

    def entropy_based_clustering(self, signals: List[NDArray],
                                n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform entropy-based clustering of signals.

        Args:
            signals: List of signal arrays
            n_clusters: Number of clusters to create

        Returns:
            Dictionary containing clustering results
        """
        try:
            if not signals or len(signals) < n_clusters:
                return {
                    'clusters': [],
                    'cluster_centers': [],
                    'cluster_labels': [],
                    'cluster_entropies': []
                }

            # Calculate entropy for each signal
            signal_entropies = []
            for signal in signals:
                if len(signal) > 0:
                    signal_entropy = self._calculate_local_entropy(signal)
                    signal_entropies.append(signal_entropy)
                else:
                    signal_entropies.append(0.0)

            signal_entropies = np.array(signal_entropies)

            # Perform hierarchical clustering
            if len(signal_entropies) > 1:
                # Calculate distance matrix
                distances = pdist(signal_entropies.reshape(-1, 1))
                linkage_matrix = linkage(distances, method='ward')

                # Perform clustering
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                # Calculate cluster centers
                cluster_centers = []
                for i in range(1, n_clusters + 1):
                    cluster_mask = cluster_labels == i
                    if np.any(cluster_mask):
                        cluster_center = np.mean(signal_entropies[cluster_mask])
                        cluster_centers.append(float(cluster_center))
                    else:
                        cluster_centers.append(0.0)

                # Group signals by cluster
                clusters = [[] for _ in range(n_clusters)]
                for i, label in enumerate(cluster_labels):
                    cluster_idx = label - 1  # Convert to 0-based indexing
                    if cluster_idx < len(clusters):
                        clusters[cluster_idx].append(signals[i])

            else:
                # Single signal case
                cluster_labels = np.array([1])
                cluster_centers = [float(signal_entropies[0])]
                clusters = [signals]

            return {
                'clusters': clusters,
                'cluster_centers': cluster_centers,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_entropies': signal_entropies.tolist()
            }

        except Exception as e:
            logger.error(f"Entropy-based clustering failed: {e}")
            return {
                'clusters': [],
                'cluster_centers': [],
                'cluster_labels': [],
                'cluster_entropies': []
            }

    def calculate_entropy_gradient(self, signal: NDArray,
                                 window_size: int = 10) -> NDArray:
        """
        Calculate entropy gradient over signal.

        Args:
            signal: Input signal array
            window_size: Window size for local entropy calculation

        Returns:
            Entropy gradient array
        """
        try:
            if len(signal) < window_size:
                return np.zeros_like(signal)

            # Calculate local entropy for each point
            entropy_values = np.zeros(len(signal))

            for i in range(len(signal)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(signal), i + window_size // 2 + 1)
                local_window = signal[start_idx:end_idx]
                entropy_values[i] = self._calculate_local_entropy(local_window)

            # Calculate gradient
            gradient = np.gradient(entropy_values)

            return gradient

        except Exception as e:
            logger.error(f"Entropy gradient calculation failed: {e}")
            return np.zeros_like(signal)

    def detect_entropy_regime_changes(self, signal: NDArray,
                                    threshold: float = 0.1) -> List[int]:
        """
        Detect regime changes in entropy patterns.

        Args:
            signal: Input signal array
            threshold: Threshold for regime change detection

        Returns:
            List of regime change indices
        """
        try:
            if len(signal) < 20:
                return []

            # Calculate entropy gradient
            gradient = self.calculate_entropy_gradient(signal)

            # Find points where gradient exceeds threshold
            regime_changes = []

            for i in range(1, len(gradient)):
                if abs(gradient[i]) > threshold:
                    regime_changes.append(i)

            # Remove consecutive changes (keep only the first)
            if regime_changes:
                filtered_changes = [regime_changes[0]]
                for change in regime_changes[1:]:
                    if change - filtered_changes[-1] > 5:  # Minimum separation
                        filtered_changes.append(change)
                regime_changes = filtered_changes

            return regime_changes

        except Exception as e:
            logger.error(f"Entropy regime change detection failed: {e}")
            return []

    def _calculate_local_entropy(self, data: NDArray) -> float:
        """Calculate local entropy of data array."""
        try:
            if len(data) == 0:
                return 0.0

            # Normalize data to probability distribution
            data_norm = data - np.min(data)
            if np.sum(data_norm) == 0:
                return 0.0

            prob_dist = data_norm / np.sum(data_norm)
            prob_dist = prob_dist[prob_dist > 0]  # Remove zeros

            if len(prob_dist) == 0:
                return 0.0

            # Calculate entropy
            entropy_val = entropy(prob_dist)

            # Normalize to [0, 1] range
            max_entropy = np.log(len(prob_dist))
            if max_entropy > 0:
                normalized_entropy = entropy_val / max_entropy
            else:
                normalized_entropy = 0.0

            return float(normalized_entropy)

        except Exception:
            return 0.0


# Global instance for convenience
entropy_engine = EntropyEngine()

# Convenience functions
def entropy_filter(signal: NDArray, threshold: float = 0.5) -> NDArray:
    """Convenience function for entropy filtering."""
    return entropy_engine.entropy_filter(signal, threshold)


def calculate_dynamic_entropy(signal: NDArray, window: int = 20) -> float:
    """Convenience function for dynamic entropy calculation."""
    return entropy_engine.calculate_dynamic_entropy(signal, window)


def entropy_wave_detection(signal: NDArray,
                         min_peak_distance: int = 5) -> Dict[str, Any]:
    """Convenience function for entropy wave detection."""
    return entropy_engine.entropy_wave_detection(signal, min_peak_distance)


def entropy_pattern_analysis(signal: NDArray,
                           pattern_length: int = 10) -> Dict[str, Any]:
    """Convenience function for entropy pattern analysis."""
    return entropy_engine.entropy_pattern_analysis(signal, pattern_length)


def entropy_based_clustering(signals: List[NDArray],
                           n_clusters: int = 3) -> Dict[str, Any]:
    """Convenience function for entropy-based clustering."""
    return entropy_engine.entropy_based_clustering(signals, n_clusters)


if __name__ == "__main__":
    # Test the entropy engine
    import numpy as np

    # Import safe print for Windows compatibility
    try:
        from ...utils.windows_cli_compatibility import safe_print
    except ImportError:
        try:
            from core.utils.windows_cli_compatibility import safe_print
        except ImportError:
            def safe_print(message):
                print(message)

    def main():
        """Main function to test entropy engine and ensure proper initialization."""
        try:
            safe_print("🌊 Testing Entropy Engine")
            safe_print("=" * 40)

            # Create test signals
            signal1 = np.random.rand(100)
            signal2 = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.rand(100)
            signal3 = np.random.rand(100) * 0.1 + 0.5  # Low entropy signal

            safe_print(f"Signal 1 (Random): {signal1.shape}")
            safe_print(f"Signal 2 (Sine + Noise): {signal2.shape}")
            safe_print(f"Signal 3 (Low Entropy): {signal3.shape}")

            # Test entropy filtering
            safe_print("\n🔍 Testing Entropy Filtering:")
            filtered_signal = entropy_filter(signal1, threshold=0.5)
            safe_print(f"✅ Filtered Signal Range: [{np.min(filtered_signal):.4f}, {np.max(filtered_signal):.4f}]")

            # Test dynamic entropy
            safe_print("\n⚡ Testing Dynamic Entropy:")
            dynamic_entropy = calculate_dynamic_entropy(signal1, window=20)
            safe_print(f"✅ Dynamic Entropy: {dynamic_entropy:.4f}")

            # Test wave detection
            safe_print("\n🌊 Testing Wave Detection:")
            wave_results = entropy_wave_detection(signal2, min_peak_distance=5)
            safe_print(f"✅ Wave Count: {wave_results['wave_count']}")
            safe_print(f"✅ Wave Frequency: {wave_results['wave_frequency']:.4f}")
            safe_print(f"✅ Wave Amplitude: {wave_results['wave_amplitude']:.4f}")

            # Test pattern analysis
            safe_print("\n📊 Testing Pattern Analysis:")
            pattern_results = entropy_pattern_analysis(signal1, pattern_length=10)
            safe_print(f"✅ Dominant Pattern: {pattern_results['dominant_pattern']}")
            safe_print(f"✅ Pattern Stability: {pattern_results['pattern_stability']:.4f}")
            safe_print(f"✅ Pattern Frequencies: {pattern_results['pattern_frequencies']}")

            # Test clustering
            safe_print("\n🎯 Testing Clustering:")
            signals = [signal1, signal2, signal3]
            clustering_results = entropy_based_clustering(signals, n_clusters=3)
            safe_print(f"✅ Cluster Centers: {clustering_results['cluster_centers']}")
            safe_print(f"✅ Cluster Labels: {clustering_results['cluster_labels']}")

            # Test advanced entropy engine features
            safe_print("\n🔬 Testing Advanced Features:")

            # Test entropy gradient
            gradient = entropy_engine.calculate_entropy_gradient(signal1, window_size=10)
            safe_print(f"✅ Entropy Gradient Shape: {gradient.shape}")
            safe_print(f"✅ Gradient Range: [{np.min(gradient):.4f}, {np.max(gradient):.4f}]")

            # Test regime change detection
            regime_changes = entropy_engine.detect_entropy_regime_changes(signal1, threshold=0.1)
            safe_print(f"✅ Regime Changes: {regime_changes}")

            # Test local entropy calculation
            local_entropy = entropy_engine._calculate_local_entropy(signal1[:10])
            safe_print(f"✅ Local Entropy: {local_entropy:.4f}")

            # Test multiple signal processing
            safe_print("\n🔄 Testing Multiple Signal Processing:")
            all_signals = [signal1, signal2, signal3]
            for i, signal in enumerate(all_signals):
                entropy_val = calculate_dynamic_entropy(signal, window=20)
                filtered = entropy_filter(signal, threshold=0.5)
                safe_print(f"✅ Signal {i+1}: Entropy={entropy_val:.4f}, Filtered Range=[{np.min(filtered):.4f}, {np.max(filtered):.4f}]")

            safe_print("\n🎉 Entropy Engine tests completed successfully!")
            return True

        except Exception as e:
            safe_print(f"❌ Entropy Engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run main function
    success = main()
    import sys
    sys.exit(0 if success else 1)
