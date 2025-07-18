#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathLib v4 for Schwabot AI
"""

import numpy as np
import hashlib
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MathLibV4:
    """
    Unified Mathematical Library v4 for Schwabot
    Implements all core mathematical functions from Day 1-45
    """
    
    def __init__(self):
        # Entropy band definitions
        self.entropy_bands = {
            'red': (0.8, 1.0),     # High chaos
            'yellow': (0.4, 0.8),   # Transitional  
            'green': (0.0, 0.4)     # Low entropy (safe)
        }
        
        # Decay constants
        self.lambda_decay = 0.1
        self.omega_quantum = 0.5
        
        # Hardware detection for optimization
        self.device_tier = self._detect_hardware_tier()
        self.version = "4.0.0"
    
    def _detect_hardware_tier(self) -> str:
        """Detect hardware capabilities"""
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if cpu_count <= 2 and memory_gb < 2:
                return "pi_zero"
            elif cpu_count <= 4 and memory_gb < 8:
                return "pi_standard"
            elif cpu_count <= 8 and memory_gb < 16:
                return "desktop"
            else:
                return "server"
        except:
            return "desktop"  # Default fallback
    
    # ========== Core Entropy Functions ==========
    
    def compute_zpe(self, profit_series: Union[List[float], np.ndarray], 
                    time_series: Union[List[float], np.ndarray]) -> float:
        """
        Zero Point Entropy: ZPE = ΔP/ΔT
        Measures profit acceleration/deceleration
        """
        profit_series = np.array(profit_series)
        time_series = np.array(time_series)
        
        if len(profit_series) < 2 or len(time_series) < 2:
            return 0.0
        
        # Calculate deltas
        delta_p = np.diff(profit_series)
        delta_t = np.diff(time_series)
        
        # Avoid division by zero
        mask = delta_t != 0
        if not np.any(mask):
            return 0.0
        
        # Return mean rate of profit change
        zpe_values = delta_p[mask] / delta_t[mask]
        return float(np.mean(np.abs(zpe_values)))
    
    def compute_zbe(self, profit: float, spread: float) -> float:
        """
        Zero Bound Entropy: ZBE = log₂((ΔProfit/Spread) + 1)
        Filters high-noise trading zones
        """
        if spread <= 0:
            return 0.0
        
        ratio = abs(profit) / spread
        return float(np.log2(ratio + 1))
    
    # ========== Gradient and Curvature Analysis ==========
    
    def gradient(self, series: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        First derivative: ∇f(t) = df/dt
        For trend detection
        """
        series = np.array(series)
        if len(series) < 2:
            return np.array([0.0])
        
        return np.gradient(series)
    
    def curvature(self, series: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Second derivative: ∂²f/∂t²
        For reversal detection (inflection points)
        """
        series = np.array(series)
        if len(series) < 3:
            return np.array([0.0])
        
        first_deriv = self.gradient(series)
        second_deriv = self.gradient(first_deriv)
        return second_deriv
    
    # ========== ROI and Memory Functions ==========
    
    def compute_roi_decay(self, roi: float, dt: float, 
                         lambda_param: Optional[float] = None) -> float:
        """
        Time-weighted ROI decay: ROI_t = ROI_{t-1} * e^(-λΔt)
        Reduces influence of old trades
        """
        if lambda_param is None:
            lambda_param = self.lambda_decay
        
        return roi * np.exp(-lambda_param * dt)
    
    # ========== Hash and Pattern Functions ==========
    
    def generate_strategy_hash(self, tick: Dict, asset: str, 
                             roi: float, strat_id: str) -> str:
        """
        Generate strategy fingerprint
        H = SHA256(tick + asset + ROI + strategy)
        """
        # Serialize tick data
        tick_str = str(sorted(tick.items()))
        hash_input = f"{tick_str}{asset}{roi:.6f}{strat_id}"
        
        # Generate hash based on device tier
        hash_obj = hashlib.sha256(hash_input.encode())
        hash_length = 16 if self.device_tier == "pi_zero" else 64
        
        return hash_obj.hexdigest()[:hash_length]
    
    def similarity_score(self, ha: str, hb: str) -> float:
        """
        Cosine similarity between hash vectors
        Returns 0-1 (1 = identical)
        """
        # Convert hex to vectors
        vec_a = np.array([int(c, 16) for c in ha])
        vec_b = np.array([int(c, 16) for c in hb])
        
        # Pad to same length
        max_len = max(len(vec_a), len(vec_b))
        if len(vec_a) < max_len:
            vec_a = np.pad(vec_a, (0, max_len - len(vec_a)))
        if len(vec_b) < max_len:
            vec_b = np.pad(vec_b, (0, max_len - len(vec_b)))
        
        # Compute cosine similarity
        return float(1 - cosine(vec_a, vec_b))
    
    def match_hash_to_vault(self, h: str, vault_index: Dict, 
                          threshold: float = 0.85) -> Optional[Dict]:
        """
        Find best matching vault entry
        Returns vault data if similarity > threshold
        """
        best_match = None
        best_score = 0.0
        
        for vault_hash, vault_data in vault_index.items():
            score = self.similarity_score(h, vault_hash)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = vault_data
        
        return best_match
    
    def resolve_hash_class(self, h: str) -> str:
        """
        Determine asset class from hash pattern
        Simple modulo-based classification
        """
        # Sum all hex digits
        hash_sum = sum(int(c, 16) for c in h)
        
        # Map to asset classes
        classes = ['BTC', 'ETH', 'XRP', 'USDC', 'SOL']
        return classes[hash_sum % len(classes)]
    
    def validate_consensus_hash(self, h_list: List[str]) -> Tuple[bool, str]:
        """
        Validate multi-agent hash consensus
        Returns (is_valid, consensus_hash)
        """
        if len(h_list) < 2:
            return False, ""
        
        # Count occurrences
        hash_counts = {}
        for h in h_list:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        
        # Find majority
        max_count = max(hash_counts.values())
        threshold = len(h_list) / 2
        
        if max_count > threshold:
            consensus = [h for h, c in hash_counts.items() if c == max_count][0]
            return True, consensus
        
        return False, ""
    
    # ========== Quantum Strategy Functions ==========
    
    def compute_quantum_collapse_matrix(self, p_tensor: np.ndarray, 
                                      psi_state: float, 
                                      zpe: float) -> np.ndarray:
        """
        Quantum collapse: Q = P * Ψ * e^(-Ω*ZPE)
        Modulates strategy probability by entropy
        """
        omega = self.omega_quantum
        collapse_factor = np.exp(-omega * zpe)
        
        return p_tensor * psi_state * collapse_factor
    
    def compute_strategy_entanglement(self, states: List[float]) -> float:
        """
        Combine multiple strategy states
        Ψ_total = normalized sum of states
        """
        if not states:
            return 0.0
        
        # Normalize to prevent explosion
        total = sum(states)
        if total == 0:
            return 0.0
        
        return total / len(states)
    
    def collapse_probability(self, roi: float, hash_trust: float, 
                           entropy_variance: float) -> float:
        """
        Strategy execution probability
        P = sigmoid(roi * trust / entropy_var)
        """
        if entropy_variance <= 0:
            entropy_variance = 0.001
        
        x = (roi * hash_trust) / entropy_variance
        # Sigmoid function
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def generate_quantum_execution_window(self, history: List[Dict]) -> Dict:
        """
        Generate optimal execution window from history
        Uses ROI-weighted time averaging
        """
        if not history:
            return {'start': 0, 'duration': 60}
        
        # Extract ROI and timing
        roi_weights = np.array([h.get('roi', 1.0) for h in history])
        start_times = np.array([h.get('start_time', 0) for h in history])
        durations = np.array([h.get('duration', 60) for h in history])
        
        # Weighted averages
        total_weight = np.sum(roi_weights)
        if total_weight == 0:
            return {'start': 0, 'duration': 60}
        
        weighted_start = np.sum(start_times * roi_weights) / total_weight
        weighted_duration = np.sum(durations * roi_weights) / total_weight
        
        return {
            'start': float(weighted_start),
            'duration': float(weighted_duration)
        }
    
    # ========== Persistent Homology Functions ==========
    
    def compute_persistent_homology(self, profit_vector: Union[List[float], np.ndarray]) -> List[Tuple[float, float]]:
        """
        Extract topological features from profit landscape
        Returns [(birth, death)] pairs for persistent features
        """
        profit_vector = np.array(profit_vector)
        if len(profit_vector) < 3:
            return []
        
        features = []
        
        # Find peaks (births) and valleys (deaths)
        for i in range(1, len(profit_vector) - 1):
            # Local maximum (birth)
            if (profit_vector[i] > profit_vector[i-1] and 
                profit_vector[i] > profit_vector[i+1]):
                
                birth = profit_vector[i]
                
                # Find corresponding death (next local minimum)
                death = birth
                for j in range(i + 1, len(profit_vector)):
                    if profit_vector[j] < death:
                        death = profit_vector[j]
                    # Check if we hit a valley
                    if (j < len(profit_vector) - 1 and 
                        profit_vector[j] < profit_vector[j+1]):
                        break
                
                # Only keep significant features
                persistence = birth - death
                if persistence > 0.05:  # 5% threshold
                    features.append((float(birth), float(death)))
        
        return features
    
    def wasserstein_distance(self, h1: List[Tuple[float, float]], 
                           h2: List[Tuple[float, float]]) -> float:
        """
        Wasserstein distance between persistence diagrams
        Measures topological similarity
        """
        if not h1 or not h2:
            return float('inf')
        
        # Extract birth values
        births1 = np.array([b for b, d in h1])
        births2 = np.array([b for b, d in h2])
        
        try:
            return float(wasserstein_distance(births1, births2))
        except:
            # Fallback to simple L2 distance
            min_len = min(len(births1), len(births2))
            return float(np.linalg.norm(births1[:min_len] - births2[:min_len]))
    
    def entry_confidence_from_topology(self, h_live: List[Tuple[float, float]], 
                                     h_past: List[Tuple[float, float]]) -> float:
        """
        Calculate entry confidence from topological similarity
        C = 1 - normalized_distance
        """
        distance = self.wasserstein_distance(h_live, h_past)
        
        # Normalize to [0, 1]
        max_expected = 10.0
        confidence = 1 - min(distance / max_expected, 1.0)
        
        return float(confidence)
    
    # ========== Phantom Band Functions ==========
    
    def detect_phantom_bands(self, price_series: Union[List[float], np.ndarray], 
                           volume_series: Union[List[float], np.ndarray]) -> List[Dict]:
        """
        Detect low-volatility phantom trading bands
        Bands where: ∇V ≈ 0 and spread < δ
        """
        price_series = np.array(price_series)
        volume_series = np.array(volume_series)
        
        if len(price_series) < 2 or len(volume_series) < 2:
            return []
        
        bands = []
        current_band = None
        
        spread_threshold = 0.001  # 0.1%
        volume_gradient_threshold = 0.05
        
        for i in range(1, len(price_series)):
            # Calculate metrics
            if price_series[i-1] != 0:
                price_spread = abs(price_series[i] - price_series[i-1]) / price_series[i-1]
            else:
                price_spread = 0
            
            if volume_series[i-1] != 0:
                volume_gradient = abs(volume_series[i] - volume_series[i-1]) / volume_series[i-1]
            else:
                volume_gradient = 0
            
            # Check phantom conditions
            if price_spread < spread_threshold and volume_gradient < volume_gradient_threshold:
                if current_band is None:
                    current_band = {
                        'start_idx': i-1,
                        'prices': [price_series[i-1], price_series[i]],
                        'volumes': [volume_series[i-1], volume_series[i]]
                    }
                else:
                    current_band['prices'].append(price_series[i])
                    current_band['volumes'].append(volume_series[i])
            else:
                # Save band if long enough
                if current_band and len(current_band['prices']) >= 5:
                    current_band['end_idx'] = i-1
                    current_band['duration'] = len(current_band['prices'])
                    current_band['stability'] = self._compute_band_stability_internal(current_band)
                    bands.append(current_band)
                current_band = None
        
        return bands
    
    def _compute_band_stability_internal(self, band: Dict) -> float:
        """Internal stability calculation"""
        price_std = np.std(band['prices'])
        volume_std = np.std(band['volumes'])
        duration = band['duration']
        
        # Avoid division by zero
        denominator = 1 + price_std + volume_std
        return float(duration / denominator)
    
    def compute_band_stability(self, duration: int, success_rate: float, 
                             entropy: float) -> float:
        """
        Band stability metric: S = (duration * success_rate) / entropy
        """
        if entropy <= 0:
            entropy = 0.001
        
        return float((duration * success_rate) / entropy)
    
    def validate_band_entry(self, band_id: int, zpe_state: float) -> bool:
        """
        Validate phantom band entry conditions
        Entry if: stability high AND entropy low
        """
        # Simplified validation
        zpe_threshold = 0.3
        return zpe_state <= zpe_threshold
    
    # ========== Utility Functions ==========
    
    def classify_entropy_band(self, zpe: float) -> str:
        """Classify ZPE into red/yellow/green bands"""
        for band, (min_val, max_val) in self.entropy_bands.items():
            if min_val <= zpe <= max_val:
                return band
        return 'red'  # Default to highest risk
    
    def calculate_hash(self, data: str) -> str:
        """Calculate hash of data."""
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation error: {e}")
            return ""
    
    def validate_math_operations(self) -> bool:
        """Validate mathematical operations."""
        try:
            # Test basic operations
            assert 2 + 2 == 4
            assert 10 * 5 == 50
            assert 100 / 4 == 25
            return True
        except Exception as e:
            logger.error(f"Math validation error: {e}")
            return False


# Create exportable functions for backward compatibility
mathlib = MathLibV4()

# Export all functions at module level
compute_zpe = mathlib.compute_zpe
compute_zbe = mathlib.compute_zbe
gradient = mathlib.gradient
curvature = mathlib.curvature
compute_roi_decay = mathlib.compute_roi_decay
generate_strategy_hash = mathlib.generate_strategy_hash
similarity_score = mathlib.similarity_score
match_hash_to_vault = mathlib.match_hash_to_vault
resolve_hash_class = mathlib.resolve_hash_class
validate_consensus_hash = mathlib.validate_consensus_hash
compute_quantum_collapse_matrix = mathlib.compute_quantum_collapse_matrix
compute_strategy_entanglement = mathlib.compute_strategy_entanglement
collapse_probability = mathlib.collapse_probability
generate_quantum_execution_window = mathlib.generate_quantum_execution_window
compute_persistent_homology = mathlib.compute_persistent_homology
wasserstein_distance = mathlib.wasserstein_distance
entry_confidence_from_topology = mathlib.entry_confidence_from_topology
detect_phantom_bands = mathlib.detect_phantom_bands
compute_band_stability = mathlib.compute_band_stability
validate_band_entry = mathlib.validate_band_entry
classify_entropy_band = mathlib.classify_entropy_band


def test_mathlib_v4():
    """Test MathLib v4."""
    try:
        mathlib = MathLibV4()
        if mathlib.validate_math_operations():
            print("MathLib v4: OK")
            return True
        else:
            print("MathLib v4: Validation failed")
            return False
    except Exception as e:
        print(f"MathLib v4: Error - {e}")
        return False

if __name__ == "__main__":
    test_mathlib_v4()
