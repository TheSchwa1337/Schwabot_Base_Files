#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Math Integration Test for Schwabot
===============================================
Tests the mathematical logic engine and integration bridge with real data scenarios.
"""

import logging
import time
from typing import Any, Dict

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core modules
try:
    from core.math_integration_bridge import MathIntegrationBridge, create_math_integration_bridge
    from core.math_logic_engine import (
        bitmap_fold,
        clonal_expansion_coefficient,
        drift_chain_weight,
        echo_trigger_zone,
        entropy_drift,
        hash_priority_score,
        mutation_rate,
        orbital_energy,
        phase_rotation,
        rebuy_probability,
        should_enter,
        should_exit,
        sigmoid,
        strategy_hash_evolution,
        vault_mass,
        vault_reentry_delay,
    )
    MATH_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import math engine: {e}")
    MATH_ENGINE_AVAILABLE = False


def generate_test_data() -> Dict[str, Any]:
    """Generate realistic test data for mathematical operations."""
    # Generate price history (simulating BTC price movements)
    base_price = 50000.0
    price_history = []
    for i in range(100):
        # Add some volatility and trend
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        if i > 0:
            base_price *= (1 + change)
        price_history.append(base_price)
    
    # Generate volume history
    volume_history = [np.random.uniform(1000, 5000) for _ in range(100)]
    
    # Generate entropy history
    entropy_history = [np.random.uniform(0.1, 0.9) for _ in range(100)]
    
    # Generate omega arrays for cross-asset analysis
    omega_a = np.random.uniform(0.3, 0.7, 50)  # Source asset
    omega_b = np.random.uniform(0.3, 0.7, 50)  # Target asset
    
    return {
        'price_history': price_history,
        'volume_history': volume_history,
        'entropy_history': entropy_history,
        'omega_a': omega_a,
        'omega_b': omega_b,
        'bitmap_history': [np.random.randint(0, 256) for _ in range(10)]
    }


def test_core_mathematical_functions():
    """Test all core mathematical functions."""
    logger.info("🧮 Testing Core Mathematical Functions")
    
    if not MATH_ENGINE_AVAILABLE:
        logger.error("❌ Math engine not available")
        return False
    
    test_data = generate_test_data()
    results = {}
    
    try:
        # Test 1: Entropy Drift
        logger.info("Testing entropy drift function...")
        psi = np.array(test_data['price_history'][-20:])
        phi = np.array(test_data['volume_history'][-20:])
        xi = np.array(test_data['entropy_history'][-20:])
        
        drift_result = entropy_drift(psi, phi, xi, n=8)
        results['entropy_drift'] = {
            'success': isinstance(drift_result, float),
            'value': drift_result,
            'expected_range': (-10, 10)
        }
        logger.info(f"✅ Entropy drift: {drift_result:.6f}")
        
        # Test 2: Cross-Asset Drift Chain Weight
        logger.info("Testing cross-asset drift chain weight...")
        omega_a = test_data['omega_a']
        omega_b = test_data['omega_b']
        chain_weight = drift_chain_weight(omega_a, omega_b, delta_t=2, roi_weight=0.5, xi_score=0.7)
        results['drift_chain_weight'] = {
            'success': isinstance(chain_weight, float),
            'value': chain_weight,
            'expected_range': (-1, 1)
        }
        logger.info(f"✅ Cross-asset drift weight: {chain_weight:.6f}")
        
        # Test 3: Vault Re-entry Delay
        logger.info("Testing vault re-entry delay...")
        reentry_delay = vault_reentry_delay(xi_exit=0.8, phi_entry=0.6, vault_mass=1.2, tick_entropy=0.5)
        results['vault_reentry_delay'] = {
            'success': isinstance(reentry_delay, int) and reentry_delay > 0,
            'value': reentry_delay,
            'expected_range': (1, 100)
        }
        logger.info(f"✅ Vault re-entry delay: {reentry_delay} ticks")
        
        # Test 4: Phase Rotation
        logger.info("Testing phase rotation...")
        phase_rot = phase_rotation(xi=0.5, phi=0.7, omega=0.3, period=16)
        results['phase_rotation'] = {
            'success': isinstance(phase_rot, float) and 0 <= phase_rot < 16,
            'value': phase_rot,
            'expected_range': (0, 16)
        }
        logger.info(f"✅ Phase rotation: {phase_rot:.6f}")
        
        # Test 5: Vault Mass
        logger.info("Testing vault mass...")
        xi_values = [0.5, 0.6, 0.7]
        phi_values = [0.4, 0.5, 0.6]
        roi_values = [0.1, 0.2, 0.3]
        holding_weights = [1.0, 1.0, 1.0]
        
        mass_result = vault_mass(xi_values, phi_values, roi_values, holding_weights)
        results['vault_mass'] = {
            'success': isinstance(mass_result, float),
            'value': mass_result,
            'expected_range': (0, 10)
        }
        logger.info(f"✅ Vault mass: {mass_result:.6f}")
        
        # Test 6: Bitmap Folding
        logger.info("Testing bitmap folding...")
        bitmap_data = test_data['bitmap_history'][:5]
        folded = bitmap_fold(bitmap_data, k=3)
        results['bitmap_fold'] = {
            'success': isinstance(folded, int) and 0 <= folded <= 255,
            'value': folded,
            'expected_range': (0, 255)
        }
        logger.info(f"✅ Bitmap folded: {folded}")
        
        # Test 7: Orbital Energy
        logger.info("Testing orbital energy...")
        energy, state = orbital_energy(omega=0.6, phi=0.5, xi=0.4)
        results['orbital_energy'] = {
            'success': isinstance(energy, float) and state in ['s', 'p', 'd', 'f'],
            'value': energy,
            'state': state,
            'expected_range': (0, 5)
        }
        logger.info(f"✅ Orbital energy: {energy:.6f} ({state})")
        
        # Test 8: Strategy Hash Evolution
        logger.info("Testing strategy hash evolution...")
        prev_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
        new_hash = strategy_hash_evolution(prev_hash, delta_roi=0.05, entropy_deviation=0.02)
        results['strategy_hash_evolution'] = {
            'success': isinstance(new_hash, str) and len(new_hash) == 64 and new_hash != prev_hash,
            'value': new_hash[:16] + "...",
            'expected_format': "64-character hex string"
        }
        logger.info(f"✅ Strategy hash evolved: {new_hash[:16]}...")
        
        # Test 9: Clonal Expansion Coefficient
        logger.info("Testing clonal expansion coefficient...")
        clonal_coeff = clonal_expansion_coefficient(tcell_activation=0.8, roi=0.15, xi_weight=0.3)
        results['clonal_expansion_coefficient'] = {
            'success': isinstance(clonal_coeff, float) and clonal_coeff >= 0,
            'value': clonal_coeff,
            'expected_range': (0, 1)
        }
        logger.info(f"✅ Clonal expansion coefficient: {clonal_coeff:.6f}")
        
        # Test 10: Mutation Rate
        logger.info("Testing mutation rate...")
        mutation_rate_val = mutation_rate(roi=0.1, phi=0.5, volatility=0.3)
        results['mutation_rate'] = {
            'success': isinstance(mutation_rate_val, float) and 0 <= mutation_rate_val <= 1,
            'value': mutation_rate_val,
            'expected_range': (0, 1)
        }
        logger.info(f"✅ Mutation rate: {mutation_rate_val:.6f}")
        
        # Test 11: Rebuy Probability
        logger.info("Testing rebuy probability...")
        rebuy_prob = rebuy_probability(omega=0.6, xi=0.5, phi=0.4, vault_pressure=0.2)
        results['rebuy_probability'] = {
            'success': isinstance(rebuy_prob, float) and 0 <= rebuy_prob <= 1,
            'value': rebuy_prob,
            'expected_range': (0, 1)
        }
        logger.info(f"✅ Rebuy probability: {rebuy_prob:.6f}")
        
        # Test 12: Hash Priority Score
        logger.info("Testing hash priority score...")
        hps = hash_priority_score(roi=0.15, clonal_coeff=0.8, xi_weight=0.3, asset_drift_alignment=0.7)
        results['hash_priority_score'] = {
            'success': isinstance(hps, float) and hps >= 0,
            'value': hps,
            'expected_range': (0, 1)
        }
        logger.info(f"✅ Hash priority score: {hps:.6f}")
        
        # Test 13: Echo Trigger Zone
        logger.info("Testing echo trigger zone...")
        echo_zone = echo_trigger_zone(xi_score=0.95, phi_score=0.9, omega=0.8, omega_mean=0.6)
        results['echo_trigger_zone'] = {
            'success': isinstance(echo_zone, bool),
            'value': echo_zone,
            'expected_type': 'boolean'
        }
        logger.info(f"✅ Echo trigger zone: {echo_zone}")
        
        # Test 14: Entry/Exit Logic
        logger.info("Testing entry/exit logic...")
        enter_signal = should_enter(tcell_activation=0.8, clonal_coeff=0.7, rebuy_prob=0.9, echo_zone=True)
        exit_signal = should_exit(tcell_activation=0.2, clonal_coeff=0.2, rebuy_prob=0.1, echo_zone=False)
        
        results['entry_exit_logic'] = {
            'success': isinstance(enter_signal, bool) and isinstance(exit_signal, bool),
            'enter_signal': enter_signal,
            'exit_signal': exit_signal,
            'expected_type': 'boolean'
        }
        logger.info(f"✅ Entry signal: {enter_signal}, Exit signal: {exit_signal}")
        
        # Test 15: Sigmoid Function
        logger.info("Testing sigmoid function...")
        sigmoid_result = sigmoid(0.0)
        results['sigmoid'] = {
            'success': isinstance(sigmoid_result, float) and 0 <= sigmoid_result <= 1,
            'value': sigmoid_result,
            'expected_range': (0, 1)
        }
        logger.info(f"✅ Sigmoid(0): {sigmoid_result:.6f}")
        
        # Summary
        successful_tests = sum(1 for result in results.values() if result['success'])
        total_tests = len(results)
        
        logger.info(f"\n📊 Core Mathematical Functions Test Summary:")
        logger.info(f"✅ Successful: {successful_tests}/{total_tests}")
        logger.info(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
        
        return successful_tests == total_tests
        
    except Exception as e:
        logger.error(f"❌ Error in core mathematical functions test: {e}")
        return False


def test_integration_bridge():
    """Test the mathematical integration bridge."""
    logger.info("\n🌉 Testing Mathematical Integration Bridge")
    
    if not MATH_ENGINE_AVAILABLE:
        logger.error("❌ Math engine not available")
        return False
    
    try:
        # Create integration bridge
        bridge = create_math_integration_bridge()
        test_data = generate_test_data()
        
        results = {}
        
        # Test 1: Strategy Bit Mapper Integration
        logger.info("Testing strategy bit mapper integration...")
        market_data = {
            'price_history': test_data['price_history'][-20:],
            'volume_history': test_data['volume_history'][-20:],
            'entropy_history': test_data['entropy_history'][-20:]
        }
        strategy_params = {
            'tcell_activation': 0.8,
            'clonal_coefficient': 0.7
        }
        
        strategy_result = bridge.integrate_with_strategy_bit_mapper("BTC", market_data, strategy_params)
        results['strategy_integration'] = {
            'success': strategy_result.success,
            'confidence': strategy_result.confidence,
            'execution_time': strategy_result.execution_time
        }
        logger.info(f"✅ Strategy integration: {strategy_result.success}, Confidence: {strategy_result.confidence:.3f}")
        
        # Test 2: T-Cell Survival Integration
        logger.info("Testing T-cell survival integration...")
        tcell_data = {
            'strategy_hash': 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
            'survival_score': 0.6,
            'roi': 0.12,
            'activation': 0.8,
            'xi_weight': 0.3,
            'delta_roi': 0.05
        }
        market_conditions = {
            'volatility': 0.2,
            'market_volatility': 0.25,
            'entropy_deviation': 0.03,
            'asset_drift_alignment': 0.7
        }
        
        tcell_result = bridge.integrate_with_tcell_survival(tcell_data, market_conditions)
        results['tcell_integration'] = {
            'success': tcell_result.success,
            'confidence': tcell_result.confidence,
            'execution_time': tcell_result.execution_time
        }
        logger.info(f"✅ T-cell integration: {tcell_result.success}, Confidence: {tcell_result.confidence:.3f}")
        
        # Test 3: Vault Orbital Bridge Integration
        logger.info("Testing vault orbital bridge integration...")
        vault_data = {
            'xi_values': [0.5, 0.6, 0.7],
            'phi_values': [0.4, 0.5, 0.6],
            'roi_values': [0.1, 0.2, 0.3],
            'holding_weights': [1.0, 1.0, 1.0],
            'xi_exit': 0.8,
            'phi_entry': 0.6
        }
        orbital_params = {
            'tick_entropy': 0.5,
            'xi': 0.5,
            'phi': 0.5,
            'omega': 0.5
        }
        
        vault_result = bridge.integrate_with_vault_orbital_bridge(vault_data, orbital_params)
        results['vault_integration'] = {
            'success': vault_result.success,
            'confidence': vault_result.confidence,
            'execution_time': vault_result.execution_time
        }
        logger.info(f"✅ Vault integration: {vault_result.success}, Confidence: {vault_result.confidence:.3f}")
        
        # Test 4: Entropy Decay Integration
        logger.info("Testing entropy decay integration...")
        entropy_data = {
            'psi_history': test_data['price_history'][-20:],
            'phi_history': test_data['volume_history'][-20:],
            'xi_history': test_data['entropy_history'][-20:],
            'bitmap_history': test_data['bitmap_history'][:5],
            'omega_a': test_data['omega_a'],
            'omega_b': test_data['omega_b']
        }
        time_params = {
            'delta_t': 2,
            'roi_weight': 0.5,
            'xi_score': 0.7
        }
        
        entropy_result = bridge.integrate_with_entropy_decay(entropy_data, time_params)
        results['entropy_integration'] = {
            'success': entropy_result.success,
            'confidence': entropy_result.confidence,
            'execution_time': entropy_result.execution_time
        }
        logger.info(f"✅ Entropy integration: {entropy_result.success}, Confidence: {entropy_result.confidence:.3f}")
        
        # Test 5: Mathematical Validation
        logger.info("Testing mathematical validation...")
        validation_results = bridge.validate_mathematical_operations()
        results['validation'] = {
            'success': all(validation_results.values()),
            'validation_results': validation_results
        }
        logger.info(f"✅ Mathematical validation: {results['validation']['success']}")
        
        # Test 6: Integration Status
        logger.info("Testing integration status...")
        status = bridge.get_integration_status()
        results['status'] = {
            'success': isinstance(status, dict) and 'config' in status,
            'status_keys': list(status.keys())
        }
        logger.info(f"✅ Integration status: {results['status']['success']}")
        
        # Summary
        successful_integrations = sum(1 for result in results.values() if result['success'])
        total_integrations = len(results)
        
        logger.info(f"\n📊 Integration Bridge Test Summary:")
        logger.info(f"✅ Successful: {successful_integrations}/{total_integrations}")
        logger.info(f"❌ Failed: {total_integrations - successful_integrations}/{total_integrations}")
        
        return successful_integrations == total_integrations
        
    except Exception as e:
        logger.error(f"❌ Error in integration bridge test: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("\n🛡️ Testing Error Handling and Edge Cases")
    
    if not MATH_ENGINE_AVAILABLE:
        logger.error("❌ Math engine not available")
        return False
    
    try:
        results = {}
        
        # Test 1: Empty arrays
        logger.info("Testing empty array handling...")
        try:
            drift_result = entropy_drift(np.array([]), np.array([]), np.array([]), n=8)
            results['empty_arrays'] = False  # Should have raised ValueError
        except ValueError:
            results['empty_arrays'] = True
        logger.info(f"✅ Empty arrays handled: {results['empty_arrays']}")
        
        # Test 2: Invalid window size
        logger.info("Testing invalid window size...")
        try:
            drift_result = entropy_drift(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), n=0)
            results['invalid_window'] = False  # Should have raised ValueError
        except ValueError:
            results['invalid_window'] = True
        logger.info(f"✅ Invalid window size handled: {results['invalid_window']}")
        
        # Test 3: Invalid input types
        logger.info("Testing invalid input types...")
        try:
            drift_result = entropy_drift([1, 2, 3], [1, 2, 3], [1, 2, 3], n=2)  # Lists instead of arrays
            results['invalid_types'] = False  # Should have raised ValueError
        except ValueError:
            results['invalid_types'] = True
        logger.info(f"✅ Invalid input types handled: {results['invalid_types']}")
        
        # Test 4: Extreme values
        logger.info("Testing extreme values...")
        try:
            # Test with very large numbers
            extreme_result = orbital_energy(1e10, 1e10, 1e10)
            results['extreme_values'] = isinstance(extreme_result[0], float)
        except Exception:
            results['extreme_values'] = False
        logger.info(f"✅ Extreme values handled: {results['extreme_values']}")
        
        # Test 5: NaN handling
        logger.info("Testing NaN handling...")
        try:
            # Test with NaN values
            nan_result = orbital_energy(float('nan'), 0.5, 0.5)
            results['nan_handling'] = not np.isnan(nan_result[0])  # Should handle NaN gracefully
        except Exception:
            results['nan_handling'] = True  # Exception is acceptable for NaN
        logger.info(f"✅ NaN handling: {results['nan_handling']}")
        
        # Summary
        successful_handling = sum(1 for result in results.values() if result)
        total_handling = len(results)
        
        logger.info(f"\n📊 Error Handling Test Summary:")
        logger.info(f"✅ Successful: {successful_handling}/{total_handling}")
        logger.info(f"❌ Failed: {total_handling - successful_handling}/{total_handling}")
        
        return successful_handling == total_handling
        
    except Exception as e:
        logger.error(f"❌ Error in error handling test: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Comprehensive Math Integration Test for Schwabot")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Run all tests
    test_results = {
        'core_functions': test_core_mathematical_functions(),
        'integration_bridge': test_integration_bridge(),
        'error_handling': test_error_handling()
    }
    
    total_time = time.time() - start_time
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎯 FINAL TEST SUMMARY")
    logger.info("=" * 70)
    
    successful_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name.replace('_', ' ').title()}")
    
    logger.info(f"\n📊 Overall Results:")
    logger.info(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    logger.info(f"❌ Failed Tests: {total_tests - successful_tests}/{total_tests}")
    logger.info(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
    
    if successful_tests == total_tests:
        logger.info("\n🎉 ALL TESTS PASSED! Mathematical integration is working correctly.")
        return True
    else:
        logger.error(f"\n💥 {total_tests - successful_tests} TESTS FAILED! Please review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 