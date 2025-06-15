"""
Forever Fractal Core Logic
========================

Implements the Forever Fractal mathematical framework for recursive state tracking
and triplet matching in matrix fault resolution.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time
import unittest
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Stub implementations for missing modules
def compute_stability_index(volume_signal, volatility_map, config=None) -> float:
    """Compute stability index from volume and volatility data."""
    if not volume_signal or not volatility_map:
        return 0.0
    # Simple stability calculation
    vol_ratio = volatility_map / (volume_signal + 1e-8)
    return 1.0 - min(vol_ratio, 1.0)

def check_shell_trade_signal(volume_signal, volatility_map, config=None) -> bool:
    """Check if shell trade signal is active."""
    if not config:
        return volume_signal > 150
    threshold = config.get('alpha_v', 15.8)
    return volume_signal > threshold * 10

def is_in_site_zone(profit_pct, stability, price, trust_score) -> bool:
    """Check if parameters are in valid site zone."""
    return (profit_pct > 0.02 and stability > 0.8 and 
            price > 10000 and trust_score > 0.8)

def post_euler_phase_drift(phase: float) -> tuple:
    """Calculate post-Euler phase drift returning loss and profit shells."""
    drift = phase * 0.95 + 0.1 * np.sin(phase)
    loss_shell = -abs(drift * 0.02)  # 2% loss shell
    profit_shell = abs(drift * 0.05)  # 5% profit shell
    return loss_shell, profit_shell

@dataclass
class FractalState:
    """Represents a fractal state vector with metadata"""
    vector: List[float]
    timestamp: float
    phase: float
    entropy: float
    recursive_depth: int = 0

class ForeverFractalCore:
    """Core implementation of the Forever Fractal mathematical framework"""
    
    def __init__(self, decay_power: float = 2.0, terms: int = 50, dimension: int = 3):
        self.decay_power = decay_power
        self.terms = terms
        self.dimension = dimension
        self.state_history: List[FractalState] = []
        self.mirror_memory: Dict[Tuple[int], float] = {}
        
    def forever_fractal(self, t: float) -> float:
        """
        Generates the forever fractal signal at time t using a power-weighted sum.
        
        Args:
            t: Time parameter
            
        Returns:
            Fractal signal value
        """
        return sum(math.sin(t * (n + 1)) / ((n + 1) ** self.decay_power) 
                  for n in range(self.terms))
    
    def generate_fractal_vector(self, t: float, phase_shift: float = 0.0) -> List[float]:
        """
        Builds a fractal state vector across multiple dimensions.
        Each dimension is a phase-shifted instance of forever_fractal.
        
        Args:
            t: Time parameter
            phase_shift: Optional phase shift
            
        Returns:
            Fractal state vector
        """
        return [self.forever_fractal(t + i * 0.5 + phase_shift) 
                for i in range(self.dimension)]
    
    def quantize_vector(self, vector: List[float], precision: float = 0.001) -> Tuple[int]:
        """
        Quantizes a fractal vector into discrete lattice points.
        
        Args:
            vector: Input vector
            precision: Quantization precision
            
        Returns:
            Quantized vector tuple
        """
        return tuple(int(x / precision) for x in vector)
    
    def compute_triplet_distance(self, a: Tuple[int], b: Tuple[int]) -> float:
        """
        Computes Euclidean distance between two quantized vectors.
        
        Args:
            a, b: Quantized vectors
            
        Returns:
            Distance metric
        """
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    
    def match_triplet(self, current: Tuple[int], memory: List[Tuple[int]], 
                     epsilon: float) -> bool:
        """
        Checks if current triplet matches any in memory within epsilon.
        
        Args:
            current: Current triplet
            memory: Historical triplets
            epsilon: Match threshold
            
        Returns:
            True if match found
        """
        return any(self.compute_triplet_distance(current, m) < epsilon 
                  for m in memory)
    
    def register_mirror(self, triplet: Tuple[int]) -> None:
        """
        Registers a mirrored version of a triplet.
        
        Args:
            triplet: Triplet to mirror
        """
        mirror_key = tuple(reversed(triplet))
        self.mirror_memory[mirror_key] = time.time()
    
    def check_mirror(self, triplet: Tuple[int], max_age: float = 3600) -> bool:
        """
        Checks if a mirrored version of the triplet exists.
        
        Args:
            triplet: Triplet to check
            max_age: Maximum age of mirror in seconds
            
        Returns:
            True if recent mirror found
        """
        mirror_key = tuple(reversed(triplet))
        if mirror_key in self.mirror_memory:
            mirror_time = self.mirror_memory[mirror_key]
            if time.time() - mirror_time < max_age:
                return True
        return False
    
    def store_state(self, vector: List[float], phase: float, entropy: float) -> None:
        """
        Stores a fractal state in history.
        
        Args:
            vector: State vector
            phase: Phase angle
            entropy: Entropy value
        """
        state = FractalState(
            vector=vector,
            timestamp=time.time(),
            phase=phase,
            entropy=entropy
        )
        self.state_history.append(state)
    
    def get_recent_states(self, count: int = 3) -> List[FractalState]:
        """
        Gets the most recent fractal states.
        
        Args:
            count: Number of states to retrieve
            
        Returns:
            List of recent states
        """
        return self.state_history[-count:]
    
    def compute_coherence(self, states: List[FractalState]) -> float:
        """
        Computes coherence score between states.
        
        Args:
            states: States to compare
            
        Returns:
            Coherence score [0,1]
        """
        if len(states) < 2:
            return 1.0
            
        # Compute phase differences
        phase_diffs = [abs(states[i].phase - states[i+1].phase) 
                      for i in range(len(states)-1)]
        
        # Compute entropy differences
        entropy_diffs = [abs(states[i].entropy - states[i+1].entropy)
                        for i in range(len(states)-1)]
        
        # Combine into coherence score
        coherence = 1.0 - (
            0.5 * sum(phase_diffs) / (2 * np.pi * (len(states)-1)) +
            0.5 * sum(entropy_diffs) / (len(states)-1)
        )
        
        return max(0.0, min(1.0, coherence))

class TestForeverFractalCore(unittest.TestCase):
    def setUp(self):
        self.core = ForeverFractalCore()

    def test_forever_fractal(self):
        t = 0.5
        expected_signal = sum(math.sin(t * (n + 1)) / ((n + 1) ** 2) 
                              for n in range(50))
        signal = self.core.forever_fractal(t)
        self.assertAlmostEqual(signal, expected_signal, places=6)

    def test_generate_fractal_vector(self):
        t = 0.5
        phase_shift = 0.1
        expected_vector = [self.core.forever_fractal(t + i * 0.5 + phase_shift) 
                            for i in range(3)]
        vector = self.core.generate_fractal_vector(t, phase_shift)
        self.assertAlmostEqual(vector, expected_vector, places=6)

    def test_quantize_vector(self):
        vector = [1.234, 2.567, 3.890]
        precision = 0.1
        expected_quantized = (1, 2, 4)
        quantized = self.core.quantize_vector(vector, precision)
        self.assertEqual(quantized, expected_quantized)

    def test_compute_triplet_distance(self):
        a = (1, 2, 3)
        b = (1.5, 2.5, 3.5)
        expected_distance = ((1 - 1.5) ** 2 + (2 - 2.5) ** 2 + (3 - 3.5) ** 2) ** 0.5
        distance = self.core.compute_triplet_distance(a, b)
        self.assertAlmostEqual(distance, expected_distance, places=6)

    def test_match_triplet(self):
        current = (1, 2, 3)
        memory = [(1, 2, 4), (2, 3, 5)]
        epsilon = 0.1
        self.assertTrue(self.core.match_triplet(current, memory, epsilon))

    def test_register_mirror(self):
        triplet = (1, 2, 3)
        self.core.register_mirror(triplet)
        self.assertIn(tuple(reversed(triplet)), self.core.mirror_memory)

    def test_check_mirror(self):
        triplet = (1, 2, 3)
        self.core.register_mirror(triplet)
        self.assertTrue(self.core.check_mirror(triplet))

    def test_store_state(self):
        vector = [0.5, 0.6, 0.7]
        phase = math.pi / 4
        entropy = 1.2
        self.core.store_state(vector, phase, entropy)
        state = self.core.state_history[-1]
        self.assertEqual(state.vector, vector)
        self.assertAlmostEqual(state.phase, phase, places=6)
        self.assertAlmostEqual(state.entropy, entropy, places=6)

    def test_get_recent_states(self):
        states = [FractalState([0.5, 0.6, 0.7], time.time(), math.pi / 4, 1.2),
                  FractalState([0.8, 0.9, 1.0], time.time() - 10, math.pi / 3, 1.1)]
        self.core.state_history = states
        recent_states = self.core.get_recent_states()
        self.assertEqual(len(recent_states), 2)
        self.assertEqual(recent_states[0].vector, [0.5, 0.6, 0.7])
        self.assertAlmostEqual(recent_states[0].phase, math.pi / 4, places=6)

    def test_compute_coherence(self):
        states = [
            FractalState([0.5, 0.6, 0.7], time.time(), math.pi / 4, 1.2),
            FractalState([0.8, 0.9, 1.0], time.time() - 10, math.pi / 3, 1.1)
        ]
        coherence = self.core.compute_coherence(states)
        self.assertAlmostEqual(coherence, 0.5, places=6)

class RecursiveSpectrumTracker:
    def __init__(self, log_dir="logs/spectrum", entropy_threshold=4.0):
        os.makedirs(log_dir, exist_ok=True)
        self.entropy_log_path = os.path.join(log_dir, "spectral_entropy_log.json")
        self.drift_map_path = os.path.join(log_dir, "fft_drift_map.npy")
        self.flattening_alerts_path = os.path.join(log_dir, "flattening_alerts.log")
        self.plot_path = os.path.join(log_dir, "rec_spectrum_plot.png")

        self.entropy_threshold = entropy_threshold
        self.previous_spectrum = None
        self.entropy_log = []
        self.spectrum_history = []
        self.dormant_threshold = 0.7  # Threshold for dormant state detection
        self.resolution_levels = [1, 2, 4, 8]  # Multi-resolution analysis levels
        self.dormant_state = False
        self.last_dormant_trigger = None
        self.harmonic_frequencies = []

    def compute_entropy(self, spectrum):
        power = np.abs(spectrum)
        power /= power.sum() + 1e-10
        entropy = -np.sum(power * np.log2(power + 1e-10))
        return entropy

    def detect_harmonic_signals(self, spectrum):
        """Detect harmonic signals in the spectrum"""
        # Find peaks in the spectrum
        peaks = np.where(np.abs(spectrum) > np.mean(np.abs(spectrum)) + 2*np.std(np.abs(spectrum)))[0]
        
        # Calculate frequency ratios between peaks
        if len(peaks) >= 2:
            ratios = []
            for i in range(len(peaks)-1):
                ratio = peaks[i+1] / peaks[i]
                if 0.9 <= ratio <= 1.1:  # Allow 10% tolerance
                    ratios.append(ratio)
            
            # If we have consistent ratios, we have a harmonic signal
            if ratios and np.std(ratios) < 0.1:
                self.harmonic_frequencies = peaks
                return True
        return False

    def check_dormant_state(self, entropy, entropy_derivative):
        """Check if system should enter dormant state"""
        if entropy < self.dormant_threshold and entropy_derivative < 0:
            if not self.dormant_state:
                self.dormant_state = True
                self.last_dormant_trigger = time.time()
                return True
        else:
            self.dormant_state = False
        return False

    def process_vector(self, vector):
        spectrum = np.fft.rfft(vector)
        entropy = self.compute_entropy(spectrum)

        now = datetime.utcnow().isoformat()
        self.entropy_log.append({"time": now, "entropy": entropy})
        self.spectrum_history.append(spectrum)

        # Multi-resolution analysis
        for level in self.resolution_levels:
            downsampled = np.array(vector)[::level]
            res_spectrum = np.fft.rfft(downsampled)
            res_entropy = self.compute_entropy(res_spectrum)
            
            # Log resolution-specific metrics
            self.entropy_log[-1][f"entropy_level_{level}"] = res_entropy

        # Check for entropy flattening
        if entropy < self.entropy_threshold:
            with open(self.flattening_alerts_path, "a") as f:
                f.write(f"[{now}] Entropy flattening detected: {entropy:.4f}\n")

        # Compare with previous spectrum
        if self.previous_spectrum is not None:
            correlation = np.correlate(np.abs(spectrum), np.abs(self.previous_spectrum), mode='valid')
            if correlation[0] > 0.99:  # Near-identical
                with open(self.flattening_alerts_path, "a") as f:
                    f.write(f"[{now}] Recursive echo trap suspected.\n")

        # Check for harmonic signals
        if self.detect_harmonic_signals(spectrum):
            with open(self.flattening_alerts_path, "a") as f:
                f.write(f"[{now}] Harmonic signal detected at frequencies: {self.harmonic_frequencies}\n")

        # Calculate entropy derivative
        if len(self.entropy_log) > 1:
            entropy_derivative = entropy - self.entropy_log[-2]["entropy"]
            
            # Check for dormant state
            if self.check_dormant_state(entropy, entropy_derivative):
                with open(self.flattening_alerts_path, "a") as f:
                    f.write(f"[{now}] Dormant state triggered. Entropy: {entropy:.4f}, Derivative: {entropy_derivative:.4f}\n")

        self.previous_spectrum = spectrum

    def save_logs(self):
        with open(self.entropy_log_path, "w") as f:
            json.dump(self.entropy_log, f, indent=2)
        np.save(self.drift_map_path, np.array([np.abs(s) for s in self.spectrum_history]))

    def plot_spectrum_heatmap(self):
        spectra = np.array([np.abs(s) for s in self.spectrum_history])
        plt.figure(figsize=(12, 6))
        plt.imshow(spectra.T, aspect='auto', cmap='magma', origin='lower')
        plt.colorbar(label="Amplitude")
        plt.title("Recursive Spectrum Evolution")
        plt.xlabel("Time Step")
        plt.ylabel("Frequency Bin")
        plt.savefig(self.plot_path)
        plt.close()

    def get_dormant_metrics(self):
        """Get metrics related to dormant state"""
        return {
            "is_dormant": self.dormant_state,
            "last_trigger": self.last_dormant_trigger,
            "harmonic_frequencies": self.harmonic_frequencies,
            "current_entropy": self.entropy_log[-1]["entropy"] if self.entropy_log else None
        }

class FractalCore:
    def __init__(self, decay_rate: float = 0.5):
        self.decay_rate = decay_rate
        self.spectrum_tracker = RecursiveSpectrumTracker()
        self.dormant_cycles = []
        self.last_collapse_time = None
        self.collapse_threshold = 0.85
        self.harmonic_boot_frequency = 0.125  # Hz
        self.recursive_depth = 0
        self.max_recursive_depth = 3
        
        # Euler-based trigger parameters
        self.euler_phase = 0.0
        self.euler_identity = np.exp(1j * np.pi) + 1
        self.post_euler_field = None
        
        # Braided mechanophore parameters
        self.braid_group = []
        self.simplicial_set = {}
        self.non_associative_state = None
        
        # Cyclic number parameters
        self.cyclic_base = 998001  # 999²
        self.cyclic_patterns = {}
        self.pattern_reversal_key = None

    def compute_post_euler_field(self, phase_drift: float) -> np.ndarray:
        """
        Compute post-Euler field equation:
        Ψ_AEON(x,t) = e^(i(π + Δθ)) * Φ(x,t) + H_Zygot(ΔS) * Λ_Bloom(x,t)
        """
        memory_shell = self._compute_memory_shell()
        entropy_reactivation = self._compute_entropy_reactivation()
        bloom_resonance = self._compute_bloom_resonance()
        
        return (np.exp(1j * (np.pi + phase_drift)) * memory_shell + 
                entropy_reactivation * bloom_resonance)
    
    def _compute_memory_shell(self) -> np.ndarray:
        """Compute memory shell Φ(x,t) from past collapse events"""
        if not self.dormant_cycles:
            return np.zeros(3)
            
        recent_cycles = self.dormant_cycles[-3:]
        return np.mean([cycle['vector'] for cycle in recent_cycles], axis=0)
    
    def _compute_entropy_reactivation(self) -> float:
        """Compute entropy reactivation H_Zygot(ΔS)"""
        if not self.spectrum_tracker.entropy_log:
            return 0.0
            
        current_entropy = self.spectrum_tracker.entropy_log[-1]['entropy']
        previous_entropy = self.spectrum_tracker.entropy_log[-2]['entropy']
        return max(0.0, current_entropy - previous_entropy)
    
    def _compute_bloom_resonance(self) -> np.ndarray:
        """Compute bloom resonance Λ_Bloom(x,t)"""
        if not self.spectrum_tracker.spectrum_history:
            return np.zeros(3)
            
        spectrum = self.spectrum_tracker.spectrum_history[-1]
        return np.abs(spectrum[:3])  # Take first 3 components
    
    def update_braid_group(self, vector: List[float]) -> None:
        """
        Update braid group representation of state evolution
        Bₙ: Used in topological quantum computation
        """
        if len(self.braid_group) >= 3:
            self.braid_group.pop(0)
        self.braid_group.append(vector)
        
        # Update simplicial set if we have enough states
        if len(self.braid_group) == 3:
            self._update_simplicial_set()
    
    def _update_simplicial_set(self) -> None:
        """Update simplicial set representation of state space"""
        if len(self.braid_group) != 3:
            return
            
        # Create simplex from braid group states
        simplex = {
            'vertices': self.braid_group,
            'edges': [
                (self.braid_group[0], self.braid_group[1]),
                (self.braid_group[1], self.braid_group[2]),
                (self.braid_group[2], self.braid_group[0])
            ],
            'face': self.braid_group
        }
        
        # Store in simplicial set
        key = hash(tuple(map(tuple, self.braid_group)))
        self.simplicial_set[key] = simplex
    
    def detect_cyclic_pattern(self, vector: List[float]) -> bool:
        """
        Detect cyclic number patterns in state vector
        Based on properties of 1/998001 (1/999²)
        """
        # Convert vector to cyclic number representation
        cyclic_value = int(abs(vector[0]) * self.cyclic_base)
        cyclic_str = str(cyclic_value).zfill(6)
        
        # Check for pattern reversal
        if cyclic_str == "998998":
            self.pattern_reversal_key = cyclic_value
            return True
            
        # Store pattern
        self.cyclic_patterns[cyclic_value] = {
            'vector': vector,
            'timestamp': time.time()
        }
        
        return False
    
    def process_recursive_state(self, vector: List[float], depth: int = 0) -> Dict[str, Any]:
        """
        Process fractal state recursively with depth control.
        Now includes Euler-based triggers, braided mechanophore structures,
        and cyclic number theory.
        """
        if depth >= self.max_recursive_depth:
            return {"status": "max_depth_reached"}

        # Update Euler phase
        self.euler_phase += 0.1
        self.post_euler_field = self.compute_post_euler_field(self.euler_phase)
        
        # Update braid group
        self.update_braid_group(vector)
        
        # Check for cyclic patterns
        cyclic_detected = self.detect_cyclic_pattern(vector)
        
        # Process through spectrum tracker
        self.spectrum_tracker.process_vector(vector)
        
        # Get dormant metrics
        dormant_metrics = self.spectrum_tracker.get_dormant_metrics()
        
        # Check for harmonic boot
        boot_detected = self._detect_harmonic_boot()
        
        # Calculate collapse risk
        collapse_risk = self._calculate_collapse_risk(vector)
        
        # Process recursively if needed
        recursive_result = None
        if depth < self.max_recursive_depth - 1:
            # Downsample vector for next level
            downsampled = vector[::2]
            recursive_result = self.process_recursive_state(downsampled, depth + 1)
        
        return {
            "depth": depth,
            "dormant_state": dormant_metrics["is_dormant"],
            "harmonic_boot": boot_detected,
            "collapse_risk": collapse_risk,
            "entropy": dormant_metrics["current_entropy"],
            "recursive_result": recursive_result,
            "euler_phase": self.euler_phase,
            "cyclic_detected": cyclic_detected,
            "braid_group_size": len(self.braid_group),
            "simplicial_set_size": len(self.simplicial_set)
        }

    def _detect_harmonic_boot(self) -> bool:
        """Detect harmonic boot signal at specified frequency"""
        if not self.spectrum_tracker.spectrum_history:
            return False

        # Get latest spectrum
        spectrum = self.spectrum_tracker.spectrum_history[-1]
        
        # Find frequency bin closest to boot frequency
        freq_bins = np.fft.rfftfreq(len(spectrum))
        boot_bin = np.argmin(np.abs(freq_bins - self.harmonic_boot_frequency))
        
        # Check if boot frequency has significant power
        boot_power = np.abs(spectrum[boot_bin])
        threshold = np.mean(np.abs(spectrum)) + 2 * np.std(np.abs(spectrum))
        
        return boot_power > threshold

    def _calculate_collapse_risk(self, vector: List[float]) -> float:
        """Calculate risk of fractal collapse"""
        if not self.spectrum_tracker.spectrum_history:
            return 0.0
            
        # Get latest spectrum
        spectrum = self.spectrum_tracker.spectrum_history[-1]
        
        # Calculate spectral coherence
        if len(self.spectrum_tracker.spectrum_history) > 1:
            prev_spectrum = self.spectrum_tracker.spectrum_history[-2]
            coherence = np.correlate(np.abs(spectrum), np.abs(prev_spectrum), mode='valid')[0]
        else:
            coherence = 1.0
            
        # Calculate entropy stability
        entropy = self.spectrum_tracker.compute_entropy(spectrum)
        entropy_stability = 1.0 - abs(entropy - self.spectrum_tracker.entropy_threshold)
        
        # Combine metrics
        collapse_risk = (1.0 - coherence) * 0.7 + (1.0 - entropy_stability) * 0.3
        
        return min(max(collapse_risk, 0.0), 1.0)

    def export_triplet_map_json(self, filepath: str):
        """
        Exports the last 5 triplet matches as structured JSON.
        
        Args:
            filepath: Path to save the JSON file
        """
        if not self.spectrum_tracker.spectrum_history:
            return
            
        # Get last 5 spectra
        recent_spectra = self.spectrum_tracker.spectrum_history[-5:]
        
        # Convert to triplets
        triplets = []
        for spectrum in recent_spectra:
            # Find dominant frequencies
            peaks = np.where(np.abs(spectrum) > np.mean(np.abs(spectrum)) + np.std(np.abs(spectrum)))[0]
            if len(peaks) >= 3:
                triplet = peaks[:3].tolist()
                triplets.append({
                    "frequencies": triplet,
                    "magnitudes": np.abs(spectrum)[peaks[:3]].tolist(),
                    "entropy": self.spectrum_tracker.compute_entropy(spectrum)
                })
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump({
                "triplets": triplets,
                "timestamp": datetime.utcnow().isoformat(),
                "dormant_state": self.spectrum_tracker.get_dormant_metrics()["is_dormant"]
            }, f, indent=2)

    def plot_entropy_drift(self, entropy_series: List[float], timestamps: List[float], label: str = "Entropy Drift"):
        """
        Plots the entropy drift over time.
        
        Args:
            entropy_series: List of entropy values
            timestamps: List of corresponding timestamps
            label: Label for the plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, entropy_series, label=label)
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        plt.title("Fractal Spectrum Entropy Drift")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def simulate_backtest(volume_signals, volatility_maps, config=None):
    results = []
    
    for volume_signal, volatility_map in zip(volume_signals, volatility_maps):
        Z = volume_signal
        N = volatility_map
        
        # Compute stability index
        S_index = compute_stability_index(Z, N, config)
        
        # Check if trade should be entered
        trade_signal = check_shell_trade_signal(Z, N, config)
        
        # Determine site zone
        is_valid_zone = is_in_site_zone(0.03, 1.0, 12000, 0.85)  # Example values
        
        # Calculate phase drift and profit shell
        loss_shell, profit_shell = post_euler_phase_drift(np.pi)
        
        results.append({
            'volume_signal': Z,
            'volatility_map': N,
            'stability_index': S_index,
            'trade_signal': trade_signal,
            'is_valid_zone': is_valid_zone,
            'loss_shell': loss_shell,
            'profit_shell': profit_shell
        })
    
    return results

# Example data
volume_signals = [100, 200, 300, 400, 500]
volatility_maps = [50, 60, 70, 80, 90]

config = {
    'alpha_v': 15.8,
    'alpha_s': 18.3,
    'alpha_c': 0.714,
    'alpha_a': 23.2,
    'alpha_p': 12.0
}

# Run backtest
backtest_results = simulate_backtest(volume_signals, volatility_maps, config)
print(backtest_results)

if __name__ == '__main__':
    unittest.main() 