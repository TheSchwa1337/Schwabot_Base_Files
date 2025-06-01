"""
Forever Fractal Integration Test Suite
====================================

Validates the integration of the Forever Fractal framework with RittleGEMM and NCCO,
ensuring mathematical soundness, pipeline stability, and profit flow integrity.
"""

import unittest
import yaml
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple
import hashlib

from core.fractal_core import ForeverFractalCore, FractalState
from core.triplet_matcher import TripletMatcher, TripletMatch
from core.matrix_fault_resolver import MatrixFaultResolver
from core.rittle_gemm import RittleGEMM
from ncco_core.ncco import NCCOCore
from schwabot.core.aleph_unitizer import AlephUnitizer

class TestFractalIntegration(unittest.TestCase):
    """Test suite for Forever Fractal integration"""
    
    @classmethod
    def setUpClass(cls):
        """Load validation configuration"""
        config_path = Path(__file__).parent.parent / 'core' / 'validation_config.yaml'
        with open(config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
            
        # Initialize components
        cls.fractal_core = ForeverFractalCore(
            decay_power=cls.config['vector_quantization']['decay_power'],
            terms=cls.config['vector_quantization']['terms'],
            dimension=cls.config['vector_quantization']['dimension']
        )
        cls.triplet_matcher = TripletMatcher(
            fractal_core=cls.fractal_core,
            epsilon=cls.config['vector_quantization']['epsilon_q'],
            min_coherence=cls.config['triplet_coherence']['min_coherence']
        )
        cls.matrix_resolver = MatrixFaultResolver()
        cls.rittle = RittleGEMM(ring_size=cls.config['integration']['rittle']['ring_size'])
        cls.ncco = NCCOCore()
        cls.aleph_unitizer = AlephUnitizer(fractal_core=cls.fractal_core)
        
    def test_vector_quantization(self):
        """Test vector quantization integrity"""
        samples = self.config['test_suites']['vector_validation']['samples']
        vectors = []
        
        # Generate test vectors
        for i in range(samples):
            t = time.time() + i * 0.1
            vector = self.fractal_core.generate_fractal_vector(t)
            vectors.append(vector)
            
        # Check lattice spacing
        for i in range(len(vectors)-1):
            for j in range(i+1, len(vectors)):
                diff = np.linalg.norm(np.array(vectors[i]) - np.array(vectors[j]))
                self.assertGreater(
                    diff,
                    self.config['vector_quantization']['epsilon_q'],
                    "Vector quantization violates minimum spacing"
                )
                
        # FFT check if enabled
        if self.config['test_suites']['vector_validation']['fft_check']:
            for vector in vectors:
                fft = np.fft.fft(vector)
                # Check for aliasing
                self.assertLess(
                    np.max(np.abs(fft[1:])),
                    np.abs(fft[0]),
                    "FFT shows aliasing in quantized states"
                )
                
    def test_triplet_coherence(self):
        """Test triplet coherence rules"""
        # Generate synthetic triplets
        triplets = []
        for _ in range(self.config['test_suites']['triplet_matching']['synthetic_triplets']):
            states = []
            for _ in range(3):
                t = time.time()
                vector = self.fractal_core.generate_fractal_vector(t)
                state = FractalState(
                    vector=vector,
                    timestamp=t,
                    phase=np.random.uniform(0, 2*np.pi),
                    entropy=np.random.uniform(0, 1)
                )
                states.append(state)
            triplets.append(states)
            
        # Test coherence calculation
        for triplet in triplets:
            match = self.triplet_matcher.find_matching_triplet(triplet)
            if match:
                self.assertGreaterEqual(
                    match.coherence,
                    self.config['triplet_coherence']['min_coherence'],
                    "Triplet coherence below threshold"
                )
                
    def test_profit_flow(self):
        """Test profit flow integrity"""
        # Initialize profit tracking
        bucket_map = {}
        last_profit = 0
        
        # Simulate profit flow
        for _ in range(self.config['test_suites']['profit_flow']['simulation_time']):
            # Generate fractal state
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            
            # Calculate profit delta
            profit_delta = np.sum(vector) - last_profit
            last_profit = np.sum(vector)
            
            # Hash to bucket
            bucket_id = hashlib.sha256(str(vector).encode()).hexdigest()[:8]
            
            # Update bucket
            if bucket_id not in bucket_map:
                bucket_map[bucket_id] = 0
            bucket_map[bucket_id] += profit_delta
            
            # Check profit allocation rules
            self.assertGreaterEqual(
                abs(profit_delta),
                self.config['profit_allocation']['min_delta'],
                "Profit delta below minimum threshold"
            )
            
        # Check for echo patterns
        echo_count = 0
        for bucket_id, profit in bucket_map.items():
            if abs(profit) < self.config['profit_allocation']['min_delta']:
                echo_count += 1
                
        self.assertLess(
            echo_count,
            self.config['profit_allocation']['echo_threshold'],
            "Too many profit echo patterns detected"
        )
        
    def test_recursive_stability(self):
        """Test recursive propagation stability"""
        vectors = []
        entropy_history = []
        
        for _ in range(self.config['test_suites']['recursive_stability']['stress_test_vectors']):
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            vectors.append(vector)
            entropy = -np.sum(np.abs(vector) * np.log2(np.abs(vector) + 1e-10))
            entropy_history.append(entropy)
            
        # Check entropy increase
        for i in range(1, len(entropy_history)):
            entropy_diff = entropy_history[i] - entropy_history[i-1]
            self.assertGreaterEqual(
                entropy_diff,
                self.config['recursive_propagation']['min_entropy_increase'],
                "Entropy not increasing in recursive propagation"
            )
            
        # Spectral analysis if enabled
        if self.config['test_suites']['recursive_stability']['spectral_analysis']:
            for vector in vectors:
                spectrum = np.fft.fft(vector)
                # Check for spectral evolution
                self.assertGreater(
                    np.std(np.abs(spectrum)),
                    0,
                    "Spectral analysis shows no evolution"
                )
                
    def test_recursive_entropy_evolution(self):
        """Test recursive entropy propagation and spectrum evolution"""
        vectors = []
        entropy_history = []

        for _ in range(self.config['test_suites']['recursive_stability']['stress_test_vectors']):
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            vectors.append(vector)
            entropy = -np.sum(np.abs(vector) * np.log2(np.abs(vector) + 1e-10))
            entropy_history.append(entropy)

        for i in range(1, len(entropy_history)):
            entropy_diff = entropy_history[i] - entropy_history[i - 1]
            self.assertGreaterEqual(
                entropy_diff,
                self.config['recursive_propagation']['min_entropy_increase'],
                "Entropy not increasing in recursive propagation"
            )

        if self.config['test_suites']['recursive_stability']['spectral_analysis']:
            for vector in vectors:
                spectrum = np.fft.fft(vector)
                self.assertGreater(
                    np.std(np.abs(spectrum)),
                    0,
                    "Spectral analysis shows no evolution"
                )
                
    def test_rittle_integration(self):
        """Test integration with RittleGEMM"""
        # Generate test data
        profit = 0.02
        ret = 0.01
        vol = 0.5
        drift = 0.1
        exec_profit = 0.015
        rebuy = 1
        price = 100.0
        
        # Update RittleGEMM
        try:
            self.rittle.update(profit, ret, vol, drift, exec_profit, rebuy, price)
        except Exception as e:
            self.fail(f"RittleGEMM update failed: {e}")
        
        # Check ring updates
        self.assertGreater(
            self.rittle.get('R1')[-1],
            self.config['integration']['rittle']['profit_threshold'],
            "RittleGEMM profit below threshold"
        )
        
        # Check volume allocation
        activations = [0.1] * 10
        allocated = self.rittle.allocate_volume(activations)
        self.assertGreater(allocated, 0, "Volume allocation failed")
        
    def test_ncco_integration(self):
        """Test integration with NCCO"""
        # Generate test pattern
        pattern = np.random.rand(3)
        
        try:
            result = self.ncco.process_pattern(pattern)
        except Exception as e:
            self.fail(f"NCCO process_pattern failed: {e}")
        
        # Check coherence
        self.assertGreaterEqual(
            result['coherence'],
            self.config['integration']['ncco']['coherence_threshold'],
            "NCCO coherence below threshold"
        )
        
        # Check profit bucket
        self.assertGreaterEqual(
            result['profit_bucket'],
            self.config['integration']['ncco']['profit_bucket_size'],
            "NCCO profit bucket below threshold"
        )

    def test_entropy_drift_slope(self):
        """Check entropy directional drift consistency"""
        drift_window = []
        for _ in range(16):
            vector = self.fractal_core.generate_fractal_vector(time.time())
            entropy = -np.sum(np.abs(vector) * np.log2(np.abs(vector) + 1e-10))
            drift_window.append(entropy)
            time.sleep(0.05)  # simulate tick delta

        # Compute first-order slope
        drift = np.gradient(drift_window)
        self.assertTrue(any(d > 0 for d in drift), "Entropy not drifting upward in cycle")

    def test_alif_fractal_integration(self):
        """Test integration between ALIF unitizer and fractal core"""
        # Generate test data
        timestamp = time.time()
        price = 100.0
        
        # Test fractal integration
        integration_result = self.aleph_unitizer.integrate_with_fractal(price, timestamp)
        
        # Verify integration metrics
        self.assertIn('fractal_entropy', integration_result)
        self.assertIn('fractal_coherence', integration_result)
        self.assertIn('fractal_vector', integration_result)
        self.assertIn('integrated_entropy', integration_result)
        self.assertIn('pattern_vector', integration_result)
        
        # Check entropy alignment
        self.assertGreater(integration_result['fractal_entropy'], 0)
        self.assertGreater(integration_result['integrated_entropy'], 0)
        
        # Verify pattern vector conversion
        self.assertEqual(len(integration_result['pattern_vector']), 8)
        self.assertTrue(all(0 <= x <= 255 for x in integration_result['pattern_vector']))

    def test_ferrous_alignment(self):
        """Test ferrous management system alignment"""
        # Generate test data
        timestamp = time.time()
        price = 100.0
        
        # Test ferrous alignment
        alignment_result = self.aleph_unitizer.check_ferrous_alignment(price, timestamp)
        
        # Verify alignment metrics
        self.assertIn('alignment_score', alignment_result)
        self.assertIn('wheel_aligned', alignment_result)
        self.assertIn('fractal_depth', alignment_result)
        self.assertIn('unitizer_depth', alignment_result)
        
        # Check alignment score bounds
        self.assertGreaterEqual(alignment_result['alignment_score'], 0)
        self.assertLessEqual(alignment_result['alignment_score'], 1)
        
        # Verify depth consistency
        self.assertGreaterEqual(alignment_result['fractal_depth'], 0)
        self.assertGreaterEqual(alignment_result['unitizer_depth'], 0)

    def test_recursive_propagation(self):
        """Test recursive propagation through ALIF and fractal systems"""
        # Generate test data
        timestamp = time.time()
        price = 100.0
        
        # Test recursive integration
        integration_result = self.aleph_unitizer.integrate_with_fractal(price, timestamp)
        alignment_result = self.aleph_unitizer.check_ferrous_alignment(price, timestamp)
        
        # Verify recursive depth consistency
        self.assertEqual(
            alignment_result['fractal_depth'],
            self.fractal_core.get_recent_states(1)[0].recursive_depth
        )
        
        # Check entropy propagation
        self.assertAlmostEqual(
            integration_result['fractal_entropy'],
            self.fractal_core.compute_entropy(integration_result['fractal_vector']),
            places=6
        )

if __name__ == '__main__':
    unittest.main() 