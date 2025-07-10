#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical Math Integration Test for Schwabot
===========================================
Demonstrates the most important mathematical improvements:
• Entropy Drift Function (𝓓) - Unified volatility analysis
• Orbital Energy Quantization - Phase class logic
• Bitmap Hash Folding - Recursive memory compression
• Cross-asset drift mapping
• Time-synchronized correlations
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

from core.bitmap_hash_folding import BitmapHashFolding, FoldedHashResult, FoldingMode

# Import the critical math systems
from core.entropy_drift_engine import DriftMode, DriftResult, EntropyDriftEngine
from core.orbital_energy_quantizer import OrbitalEnergyQuantizer, OrbitalEnergyResult, OrbitalState
from core.strategy_bit_mapper import ExpansionMode, StrategyBitMapper
from core.symbolic_registry import SymbolicRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CriticalMathIntegrationTest:
    """Test of critical mathematical improvements."""
    
    def __init__(self):
        """Initialize test systems."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize critical math systems
        self.entropy_drift = EntropyDriftEngine()
        self.orbital_quantizer = OrbitalEnergyQuantizer()
        self.bitmap_folding = BitmapHashFolding()
        self.strategy_mapper = StrategyBitMapper()
        self.symbolic_registry = SymbolicRegistry()
        
        self.logger.info("✅ Critical Math Integration Test initialized")
    
    def test_entropy_drift_function(self):
        """Test the unified entropy drift function (𝓓)."""
        self.logger.info("\n🔬 Testing Entropy Drift Function (𝓓)")
        self.logger.info("𝓓(t) = std(ψ[t-n:t]) * (1 + ∇ψ[t]) - (Ω_mean / Φ_mean)")
        
        # Generate test signals for different assets
        test_signals = {
            'BTC': np.array([45000, 45100, 45200, 45300, 45400, 45500, 45600, 45700]),
            'ETH': np.array([3000, 3010, 3020, 3015, 3030, 3040, 3050, 3060]),
            'XRP': np.array([0.5, 0.51, 0.52, 0.51, 0.53, 0.54, 0.55, 0.56]),
            'SOL': np.array([100, 105, 110, 115, 120, 125, 130, 135]),
            'USDC': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        }
        
        # Generate phase values (Ω, Φ, Ξ)
        omega_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        phi_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        xi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        for asset, signal in test_signals.items():
            self.logger.info(f"\n--- Testing {asset} ---")
            
            # Calculate entropy drift
            drift_result = self.entropy_drift.calculate_entropy_drift(
                signal=signal,
                asset=asset,
                omega_values=omega_values,
                phi_values=phi_values,
                xi_values=xi_values,
                mode=DriftMode.STANDARD
            )
            
            # Display results
            self.logger.info(f"Drift Value: {drift_result.drift_value:.6f}")
            self.logger.info(f"Volatility Weight: {drift_result.volatility_weight:.3f}")
            self.logger.info(f"Phase Correlation: {drift_result.phase_correlation:.3f}")
            self.logger.info(f"Orbital Energy: {drift_result.orbital_energy:.3f}")
            self.logger.info(f"Confidence: {drift_result.confidence:.3f}")
            self.logger.info(f"Time Offset: {drift_result.time_offset}")
            
            # Show cross-asset drift predictions
            if drift_result.cross_asset_drift:
                self.logger.info("Cross-Asset Drift Predictions:")
                for target_asset, predicted_drift in drift_result.cross_asset_drift.items():
                    self.logger.info(f"  {asset} → {target_asset}: {predicted_drift:.6f}")
        
        # Test cross-asset movement prediction
        self.logger.info("\n--- Cross-Asset Movement Prediction ---")
        btc_drift = 0.15
        for target_asset in ['ETH', 'XRP', 'SOL']:
            predicted_drift, time_offset = self.entropy_drift.predict_cross_asset_movement(
                'BTC', target_asset, btc_drift
            )
            self.logger.info(f"BTC → {target_asset}: drift={predicted_drift:.6f}, offset={time_offset} ticks")
    
    def test_orbital_energy_quantization(self):
        """Test orbital energy quantization."""
        self.logger.info("\n🔬 Testing Orbital Energy Quantization")
        self.logger.info("orbital_energy(Ω,Ξ,Φ) = (Ω² + Φ) * log(Ξ + 1e-6)")
        
        # Test different energy scenarios
        test_scenarios = [
            {
                'name': 'Low Energy (S orbital)',
                'omega': np.array([0.1, 0.15, 0.2, 0.25]),
                'phi': np.array([0.05, 0.1, 0.15, 0.2]),
                'xi': np.array([0.01, 0.02, 0.03, 0.04])
            },
            {
                'name': 'Medium Energy (P orbital)',
                'omega': np.array([0.4, 0.45, 0.5, 0.55]),
                'phi': np.array([0.3, 0.35, 0.4, 0.45]),
                'xi': np.array([0.2, 0.25, 0.3, 0.35])
            },
            {
                'name': 'High Energy (D orbital)',
                'omega': np.array([0.7, 0.75, 0.8, 0.85]),
                'phi': np.array([0.6, 0.65, 0.7, 0.75]),
                'xi': np.array([0.5, 0.55, 0.6, 0.65])
            },
            {
                'name': 'Very High Energy (F orbital)',
                'omega': np.array([1.2, 1.25, 1.3, 1.35]),
                'phi': np.array([1.1, 1.15, 1.2, 1.25]),
                'xi': np.array([1.0, 1.05, 1.1, 1.15])
            }
        ]
        
        for scenario in test_scenarios:
            self.logger.info(f"\n--- {scenario['name']} ---")
            
            # Calculate orbital energy
            energy_result = self.orbital_quantizer.calculate_orbital_energy(
                omega_values=scenario['omega'],
                phi_values=scenario['phi'],
                xi_values=scenario['xi']
            )
            
            # Display results
            self.logger.info(f"Energy Value: {energy_result.energy_value:.6f}")
            self.logger.info(f"Orbital State: {energy_result.orbital_state.value}")
            self.logger.info(f"Phase Depth: {energy_result.phase_depth:.3f}")
            self.logger.info(f"Quantum Coherence: {energy_result.quantum_coherence:.3f}")
            self.logger.info(f"Transition Probability: {energy_result.transition_probability:.3f}")
            self.logger.info(f"Confidence: {energy_result.confidence:.3f}")
            
            # Predict next orbital state
            energy_trend = 0.1  # Positive trend
            next_state, probability = self.orbital_quantizer.predict_orbital_transition(
                energy_result.orbital_state, energy_trend
            )
            self.logger.info(f"Predicted Next State: {next_state.value} (p={probability:.3f})")
    
    def test_bitmap_hash_folding(self):
        """Test bitmap hash folding for recursive memory compression."""
        self.logger.info("\n🔬 Testing Bitmap Hash Folding")
        self.logger.info("folded_hash(t) = XOR(bitmap[t], bitmap[t-1]) + rotate_left(bitmap[t-2], 3)")
        
        # Generate test bitmaps
        test_bitmaps = []
        for i in range(8):
            # Create bitmap with some pattern and noise
            base_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 16)  # 128-bit pattern
            noise = np.random.randint(0, 2, 128)
            bitmap = np.logical_xor(base_pattern, noise).astype(np.uint8)
            test_bitmaps.append(bitmap)
        
        # Test different folding modes
        folding_modes = [
            FoldingMode.XOR_ROTATE,
            FoldingMode.LINEAR_COMBINE,
            FoldingMode.NONLINEAR_MIX,
            FoldingMode.RECURSIVE_FOLD
        ]
        
        for mode in folding_modes:
            self.logger.info(f"\n--- Testing {mode.value} Folding ---")
            
            # Clear history for clean test
            self.bitmap_folding.clear_history()
            
            # Process each bitmap
            for i, bitmap in enumerate(test_bitmaps):
                fold_result = self.bitmap_folding.fold_bitmap_hash(bitmap, mode=mode)
                
                if i >= 2:  # Only show results after we have enough history
                    self.logger.info(f"Bitmap {i+1}:")
                    self.logger.info(f"  Hash: {fold_result.hash_hex[:16]}...")
                    self.logger.info(f"  Compression Ratio: {fold_result.compression_ratio:.3f}")
                    self.logger.info(f"  Memory Footprint: {fold_result.memory_footprint} bytes")
                    self.logger.info(f"  Recursion Depth: {fold_result.recursion_depth}")
                    self.logger.info(f"  Confidence: {fold_result.confidence:.3f}")
        
        # Show folding statistics
        stats = self.bitmap_folding.get_folding_statistics()
        self.logger.info(f"\nFolding Statistics: {stats}")
    
    def test_integrated_critical_math(self):
        """Test all critical math systems working together."""
        self.logger.info("\n🔬 Testing Integrated Critical Math Workflow")
        
        # Simulate a complete trading scenario
        self.logger.info("--- Complete Trading Scenario ---")
        
        # 1. Generate market data
        btc_signal = np.array([45000, 45100, 45200, 45300, 45400, 45500, 45600, 45700])
        omega_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        phi_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        xi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # 2. Calculate entropy drift
        self.logger.info("Step 1: Entropy Drift Calculation")
        drift_result = self.entropy_drift.calculate_entropy_drift(
            signal=btc_signal,
            asset='BTC',
            omega_values=omega_values,
            phi_values=phi_values,
            xi_values=xi_values
        )
        self.logger.info(f"  Drift: {drift_result.drift_value:.6f}, Confidence: {drift_result.confidence:.3f}")
        
        # 3. Calculate orbital energy
        self.logger.info("Step 2: Orbital Energy Quantization")
        energy_result = self.orbital_quantizer.calculate_orbital_energy(
            omega_values=omega_values,
            phi_values=phi_values,
            xi_values=xi_values
        )
        self.logger.info(f"  Energy: {energy_result.energy_value:.6f}, Orbital: {energy_result.orbital_state.value}")
        
        # 4. Generate bitmap from signal
        self.logger.info("Step 3: Bitmap Generation and Folding")
        # Convert signal to bitmap (simplified)
        signal_normalized = (btc_signal - np.min(btc_signal)) / (np.max(btc_signal) - np.min(btc_signal))
        bitmap = (signal_normalized * 255).astype(np.uint8)
        
        fold_result = self.bitmap_folding.fold_bitmap_hash(bitmap)
        self.logger.info(f"  Folded Hash: {fold_result.hash_hex[:16]}...")
        self.logger.info(f"  Compression: {fold_result.compression_ratio:.3f}")
        
        # 5. Strategy classification
        self.logger.info("Step 4: Strategy Classification")
        strategy_result = self.strategy_mapper.classify_signal(
            signal=btc_signal,
            asset='BTC',
            entropy_level=drift_result.drift_value,
            tick_index=1
        )
        self.logger.info(f"  4-bit Strategy: {strategy_result.strategy_4bit}")
        self.logger.info(f"  8-bit Microstrategy: {strategy_result.strategy_8bit}")
        self.logger.info(f"  Strategy Name: {strategy_result.strategy_name}")
        
        # 6. Symbolic math integration
        self.logger.info("Step 5: Symbolic Math Integration")
        gradient_symbol = self.symbolic_registry.get_symbol('∇')
        omega_symbol = self.symbolic_registry.get_symbol('Ω')
        phi_symbol = self.symbolic_registry.get_symbol('Φ')
        
        if gradient_symbol and omega_symbol and phi_symbol:
            self.logger.info(f"  Using symbols: {gradient_symbol.symbol}, {omega_symbol.symbol}, {phi_symbol.symbol}")
            self.logger.info(f"  Gradient def: {gradient_symbol.mathematical_definition}")
            self.logger.info(f"  Omega def: {omega_symbol.mathematical_definition}")
            self.logger.info(f"  Phi def: {phi_symbol.mathematical_definition}")
        
        # 7. Final decision synthesis
        self.logger.info("Step 6: Decision Synthesis")
        
        # Combine all results for final decision
        decision_confidence = (
            drift_result.confidence * 0.3 +
            energy_result.confidence * 0.3 +
            fold_result.confidence * 0.2 +
            strategy_result.confidence * 0.2
        )
        
        # Determine action based on combined signals
        if energy_result.orbital_state == OrbitalState.F and drift_result.drift_value > 0.1:
            action = "HIGH_VOLATILITY_EXIT"
        elif energy_result.orbital_state == OrbitalState.S and drift_result.drift_value < 0.05:
            action = "STABLE_HOLD"
        elif strategy_result.strategy_4bit in [4, 10, 12, 15]:  # BTC preferences
            action = "BTC_OPTIMIZED_ENTRY"
        else:
            action = "MONITOR_WAIT"
        
        self.logger.info(f"  Final Action: {action}")
        self.logger.info(f"  Decision Confidence: {decision_confidence:.3f}")
        
        # 8. Cross-asset implications
        self.logger.info("Step 7: Cross-Asset Implications")
        for target_asset in ['ETH', 'XRP', 'SOL']:
            predicted_drift, time_offset = self.entropy_drift.predict_cross_asset_movement(
                'BTC', target_asset, drift_result.drift_value
            )
            self.logger.info(f"  BTC → {target_asset}: {predicted_drift:.6f} (offset: {time_offset} ticks)")
    
    def test_performance_metrics(self):
        """Test performance and efficiency metrics."""
        self.logger.info("\n🔬 Testing Performance Metrics")
        
        # Get statistics from all systems
        drift_stats = self.entropy_drift.get_drift_statistics()
        orbital_stats = self.orbital_quantizer.get_orbital_statistics()
        folding_stats = self.bitmap_folding.get_folding_statistics()
        
        self.logger.info("System Statistics:")
        self.logger.info(f"  Entropy Drift: {drift_stats}")
        self.logger.info(f"  Orbital Energy: {orbital_stats}")
        self.logger.info(f"  Bitmap Folding: {folding_stats}")
        
        # Test memory efficiency
        self.logger.info("\nMemory Efficiency Test:")
        
        # Generate large dataset
        large_signal = np.random.random(1000)
        large_omega = np.random.random(1000)
        large_phi = np.random.random(1000)
        large_xi = np.random.random(1000)
        
        start_time = time.time()
        
        # Process large dataset
        drift_result = self.entropy_drift.calculate_entropy_drift(
            signal=large_signal,
            asset='BTC',
            omega_values=large_omega,
            phi_values=large_phi,
            xi_values=large_xi
        )
        
        energy_result = self.orbital_quantizer.calculate_orbital_energy(
            omega_values=large_omega,
            phi_values=large_phi,
            xi_values=large_xi
        )
        
        large_bitmap = (large_signal * 255).astype(np.uint8)
        fold_result = self.bitmap_folding.fold_bitmap_hash(large_bitmap)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.logger.info(f"  Large Dataset Processing Time: {processing_time:.3f} seconds")
        self.logger.info(f"  Drift Confidence: {drift_result.confidence:.3f}")
        self.logger.info(f"  Energy Confidence: {energy_result.confidence:.3f}")
        self.logger.info(f"  Folding Confidence: {fold_result.confidence:.3f}")
    
    def run_all_tests(self):
        """Run all critical math tests."""
        self.logger.info("🚀 Starting Critical Math Integration Tests")
        self.logger.info("=" * 70)
        
        try:
            # Run individual system tests
            self.test_entropy_drift_function()
            self.test_orbital_energy_quantization()
            self.test_bitmap_hash_folding()
            
            # Run integrated workflow test
            self.test_integrated_critical_math()
            
            # Run performance test
            self.test_performance_metrics()
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("✅ All critical math tests completed successfully!")
            self.logger.info("🎯 Key Mathematical Improvements Implemented:")
            self.logger.info("  • Entropy Drift Function (𝓓) - Unified volatility analysis")
            self.logger.info("  • Orbital Energy Quantization - Phase class logic")
            self.logger.info("  • Bitmap Hash Folding - Recursive memory compression")
            self.logger.info("  • Cross-asset drift mapping - Time-synchronized correlations")
            self.logger.info("  • Symbolic math integration - Dynamic operator registry")
            
        except Exception as e:
            self.logger.error(f"❌ Test failed: {e}")
            raise

def main():
    """Main test execution."""
    test = CriticalMathIntegrationTest()
    test.run_all_tests()

if __name__ == "__main__":
    main() 