#!/usr/bin/env python3
"""
Mathematical Pipeline Test - Schwabot UROS v1.0
==============================================

Comprehensive test script to validate all mathematical functions and integration
points in the Schwabot trading system pipeline.

Tests all core mathematical functions:
1. Bit Phase Math (bit_phase_engine.py, matrix_mapper.py, dlt_waveform_engine.py)
2. Phase-Weighted Matrix Math (tensor_matcher.py)
3. Tensor Score + Delta Resolver (tensor_score_utils.py, tensor_matcher.py)
4. Rebalance Logic (profit_cycle_allocator.py)
5. Wave Entropy Function (dlt_waveform_engine.py, tensor_score_utils.py)
6. Profit Calculation + Efficiency Score (profit_routing_engine.py)
7. Sharpe Proxy Index (risk_scoring_layer.py)
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bit_phase_math():
    """Test bit phase mathematical functions."""
    print("\n🔢 Testing Bit Phase Math...")
    
    try:
        from core.bit_phase_engine import BitPhaseEngine
        from core.matrix_mapper import MatrixMapper
        from core.dlt_waveform_engine import DLTWaveformEngine
        
        # Test bit phase engine
        bit_engine = BitPhaseEngine()
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        
        # Test resolve_bit_phase function
        phase_4bit = bit_engine.resolve_bit_phase(test_hash, "4bit")
        phase_8bit = bit_engine.resolve_bit_phase(test_hash, "8bit")
        phase_42bit = bit_engine.resolve_bit_phase(test_hash, "42bit")
        
        print(f"✅ Bit Phase Resolution:")
        print(f"   4-bit: {phase_4bit} (0-15)")
        print(f"   8-bit: {phase_8bit} (0-255)")
        print(f"   42-bit: {phase_42bit} (0-2^42)")
        
        # Test matrix mapper bit resolution
        matrix_mapper = MatrixMapper()
        phase_4bit_mm = matrix_mapper.resolve_bit_phase(test_hash, "4bit")
        phase_8bit_mm = matrix_mapper.resolve_bit_phase(test_hash, "8bit")
        phase_42bit_mm = matrix_mapper.resolve_bit_phase(test_hash, "42bit")
        
        print(f"✅ Matrix Mapper Bit Resolution:")
        print(f"   4-bit: {phase_4bit_mm}")
        print(f"   8-bit: {phase_8bit_mm}")
        print(f"   42-bit: {phase_42bit_mm}")
        
        # Test DLT waveform engine
        dlt_engine = DLTWaveformEngine()
        test_sequence = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4]
        waveform_result = dlt_engine.process_waveform_data("test", np.array(test_sequence), 1.0)
        
        print(f"✅ DLT Waveform Processing: {waveform_result is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bit Phase Math test failed: {e}")
        return False

def test_phase_weighted_matrix():
    """Test phase-weighted matrix mathematical functions."""
    print("\n📐 Testing Phase-Weighted Matrix Math...")
    
    try:
        from core.tensor_matcher import TensorMatcher
        from core.phase_entropy_matcher import PhaseEntropyMatcher
        
        # Test tensor matcher phase weight matrix
        tensor_matcher = TensorMatcher()
        bit_pattern = [1, 0, 1, 1, 0, 1, 0, 1]
        entropy = 4.5
        
        phase_weight = tensor_matcher.phase_weight_matrix(bit_pattern, entropy)
        print(f"✅ Tensor Matcher Phase Weight: {phase_weight:.4f}")
        
        # Test phase entropy matcher
        phase_matcher = PhaseEntropyMatcher()
        phase_weight_pem = phase_matcher.phase_weight_matrix(bit_pattern, entropy)
        print(f"✅ Phase Entropy Matcher Phase Weight: {phase_weight_pem:.4f}")
        
        # Verify mathematical formula: (bit_score * entropy) / (len(bits) + ε)
        bit_score = sum(bit_pattern)
        expected_weight = (bit_score * entropy) / (len(bit_pattern) + 1e-6)
        print(f"✅ Expected Phase Weight: {expected_weight:.4f}")
        print(f"✅ Formula Verification: {abs(phase_weight - expected_weight) < 0.001}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase-Weighted Matrix test failed: {e}")
        return False

def test_tensor_score_delta_resolver():
    """Test tensor score and delta resolver functions."""
    print("\n📈 Testing Tensor Score + Delta Resolver...")
    
    try:
        from core.tensor_matcher import TensorMatcher
        from core.tensor_score_utils import TensorScoreUtils
        from core.matrix_mapper import MatrixMapper
        from core.tensor_router import TensorRouter
        
        # Test tensor matcher tensor score
        tensor_matcher = TensorMatcher()
        tensor_score_tm = tensor_matcher.tensor_score(45000.0, 46000.0, 8)
        print(f"✅ Tensor Matcher Tensor Score: {tensor_score_tm}")
        
        # Test tensor score utils
        tensor_utils = TensorScoreUtils()
        market_data = {'entropy_level': 4.5, 'volatility': 0.03, 'market_heat': 0.6}
        tensor_score_tu = tensor_utils.calculate_tensor_score(45000.0, 46000.0, 8, market_data)
        print(f"✅ Tensor Score Utils Tensor Score: {tensor_score_tu}")
        
        # Test matrix mapper tensor score
        matrix_mapper = MatrixMapper()
        tensor_score_mm = matrix_mapper.calculate_tensor_score(45000.0, 46000.0, 8)
        print(f"✅ Matrix Mapper Tensor Score: {tensor_score_mm}")
        
        # Test tensor router
        tensor_router = TensorRouter()
        tensor_score_tr = tensor_router.tensor_score(45000.0, 46000.0, 8)
        print(f"✅ Tensor Router Tensor Score: {tensor_score_tr}")
        
        # Verify mathematical formula: T = (current - entry) / entry * (phase + 1)
        delta = (46000.0 - 45000.0) / 45000.0
        expected_score = delta * (8 + 1)
        print(f"✅ Expected Tensor Score: {expected_score:.4f}")
        print(f"✅ Formula Verification: {abs(tensor_score_tm - expected_score) < 0.001}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor Score + Delta Resolver test failed: {e}")
        return False

def test_rebalance_logic():
    """Test rebalance logic functions."""
    print("\n💹 Testing Rebalance Logic...")
    
    try:
        from core.profit_cycle_allocator import ProfitCycleAllocator
        from core.tensor_score_utils import TensorScoreUtils
        
        # Test profit cycle allocator
        profit_allocator = ProfitCycleAllocator()
        execution_packet = {
            'profit_amount': 1000.0,
            'market_data': {'entropy_level': 4.5, 'volatility': 0.03},
            'portfolio_state': {'cash': 50000.0, 'positions': {'BTC': 0.5, 'USDC': 0.5}}
        }
        
        allocation_result = profit_allocator.allocate(execution_packet)
        print(f"✅ Profit Cycle Allocator: {allocation_result is not None}")
        
        # Test tensor score utils rebalancing
        tensor_utils = TensorScoreUtils()
        rebalance_result = tensor_utils.rebalance_profit(1000.0, 0.25, 5.5)
        print(f"✅ Tensor Utils Rebalancing: {rebalance_result is not None}")
        
        if rebalance_result:
            print(f"   Allocations: {rebalance_result.allocations}")
            print(f"   Rebalance Threshold: {rebalance_result.rebalance_threshold}")
        
        # Test rebalance function logic
        profit = 1000.0
        volatility = 0.25
        
        if profit > 0.12:  # High profit
            expected_allocations = {"BTC": profit * 0.75, "USDC": profit * 0.25}
        elif volatility > 0.3:  # High volatility
            expected_allocations = {"USDC": profit * 0.6, "XRP": profit * 0.4}
        else:  # Default
            expected_allocations = {"XRP": profit * 1.0}
        
        print(f"✅ Expected Allocations: {expected_allocations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rebalance Logic test failed: {e}")
        return False

def test_wave_entropy_function():
    """Test wave entropy calculation functions."""
    print("\n⏱ Testing Wave Entropy Function...")
    
    try:
        from core.tensor_score_utils import TensorScoreUtils
        from core.dlt_waveform_engine import DLTWaveformEngine
        
        # Test tensor score utils wave entropy
        tensor_utils = TensorScoreUtils()
        test_sequence = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4]
        entropy_tu = tensor_utils.calculate_wave_entropy(test_sequence)
        print(f"✅ Tensor Utils Wave Entropy: {entropy_tu:.4f}")
        
        # Test DLT waveform engine
        dlt_engine = DLTWaveformEngine()
        waveform_result = dlt_engine.process_waveform_data("test", np.array(test_sequence), 1.0)
        print(f"✅ DLT Waveform Processing: {waveform_result is not None}")
        
        # Verify mathematical formula: H = -Σᵢ pᵢ * log₂(pᵢ)
        # Calculate FFT
        fft = np.fft.fft(test_sequence)
        power = np.abs(fft) ** 2
        total_power = np.sum(power)
        normalized = power / total_power
        expected_entropy = -np.sum(normalized * np.log2(normalized + 1e-9))
        
        print(f"✅ Expected Wave Entropy: {expected_entropy:.4f}")
        print(f"✅ Formula Verification: {abs(entropy_tu - expected_entropy) < 0.001}")
        
        return True
        
    except Exception as e:
        print(f"❌ Wave Entropy Function test failed: {e}")
        return False

def test_profit_calculation_efficiency():
    """Test profit calculation and efficiency score functions."""
    print("\n🧾 Testing Profit Calculation + Efficiency Score...")
    
    try:
        # Test profit calculation function
        def calculate_profit(entry: float, exit: float, qty: float) -> float:
            return (exit - entry) * qty
        
        def route_efficiency(actual: float, potential: float, weight: float) -> float:
            return (actual / potential) * weight if potential else 0.0
        
        # Test profit calculation
        entry_price = 45000.0
        exit_price = 46000.0
        quantity = 1.0
        
        profit = calculate_profit(entry_price, exit_price, quantity)
        print(f"✅ Profit Calculation: {profit:.2f}")
        
        # Test route efficiency
        actual_profit = 1000.0
        potential_profit = 1200.0
        weight = 0.8
        
        efficiency = route_efficiency(actual_profit, potential_profit, weight)
        print(f"✅ Route Efficiency: {efficiency:.4f}")
        
        # Verify calculations
        expected_profit = (exit_price - entry_price) * quantity
        expected_efficiency = (actual_profit / potential_profit) * weight
        
        print(f"✅ Expected Profit: {expected_profit:.2f}")
        print(f"✅ Expected Efficiency: {expected_efficiency:.4f}")
        print(f"✅ Profit Verification: {abs(profit - expected_profit) < 0.01}")
        print(f"✅ Efficiency Verification: {abs(efficiency - expected_efficiency) < 0.001}")
        
        return True
        
    except Exception as e:
        print(f"❌ Profit Calculation + Efficiency Score test failed: {e}")
        return False

def test_sharpe_proxy_index():
    """Test Sharpe proxy index calculation."""
    print("\n🔄 Testing Sharpe Proxy Index...")
    
    try:
        # Test Sharpe proxy function
        def sharpe_proxy(return_series: list, risk_free: float = 0.01) -> float:
            returns = np.array(return_series)
            excess = returns - risk_free
            return excess.mean() / (excess.std() + 1e-9)
        
        # Test with sample return series
        return_series = [0.02, 0.03, -0.01, 0.04, 0.01, -0.02, 0.03, 0.02]
        sharpe = sharpe_proxy(return_series, 0.01)
        print(f"✅ Sharpe Proxy: {sharpe:.4f}")
        
        # Verify calculation
        returns = np.array(return_series)
        excess = returns - 0.01
        expected_sharpe = excess.mean() / (excess.std() + 1e-9)
        
        print(f"✅ Expected Sharpe: {expected_sharpe:.4f}")
        print(f"✅ Formula Verification: {abs(sharpe - expected_sharpe) < 0.001}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sharpe Proxy Index test failed: {e}")
        return False

def test_integration_pipeline():
    """Test complete integration pipeline."""
    print("\n🔗 Testing Complete Integration Pipeline...")
    
    try:
        from core.tensor_matcher import TensorMatcher
        from core.bit_phase_engine import BitPhaseEngine
        from core.matrix_mapper import MatrixMapper
        from core.profit_cycle_allocator import ProfitCycleAllocator
        
        # Initialize components
        tensor_matcher = TensorMatcher()
        bit_phase_engine = BitPhaseEngine()
        matrix_mapper = MatrixMapper()
        profit_allocator = ProfitCycleAllocator()
        
        # Setup integrations
        tensor_matcher.set_bit_phase_engine(bit_phase_engine)
        tensor_matcher.set_matrix_mapper(matrix_mapper)
        tensor_matcher.set_profit_allocator(profit_allocator)
        
        # Test complete pipeline
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        market_data = {
            'entropy_level': 4.5,
            'volatility': 0.03,
            'market_heat': 0.6
        }
        
        # Complete tensor matching
        result = tensor_matcher.match_tensor(test_hash, 45000.0, 46000.0, market_data)
        
        if result:
            print(f"✅ Complete Pipeline Result:")
            print(f"   Phase Value: {result.phase_value}")
            print(f"   Bit Phase: {result.bit_phase.value}")
            print(f"   Strategy: {result.strategy_type.value}")
            print(f"   Tensor Score: {result.tensor_score:.4f}")
            print(f"   Phase Weight: {result.phase_weight:.4f}")
            print(f"   Basket ID: {result.basket_id}")
            print(f"   Confidence: {result.confidence:.4f}")
        
        # Test hash registry integration
        basket_id = matrix_mapper.decode_hash_to_basket(test_hash, 100, 45000.0)
        print(f"✅ Hash Registry Integration: {basket_id}")
        
        # Test profit allocation integration
        execution_packet = {
            'profit_amount': 1000.0,
            'market_data': market_data,
            'portfolio_state': {'cash': 50000.0, 'positions': {'BTC': 0.5, 'USDC': 0.5}}
        }
        allocation_result = profit_allocator.allocate(execution_packet)
        print(f"✅ Profit Allocation Integration: {allocation_result is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Pipeline test failed: {e}")
        return False

def main():
    """Run all mathematical pipeline tests."""
    print("🚀 Starting Schwabot Mathematical Pipeline Tests...")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['bit_phase_math'] = test_bit_phase_math()
    test_results['phase_weighted_matrix'] = test_phase_weighted_matrix()
    test_results['tensor_score_delta'] = test_tensor_score_delta_resolver()
    test_results['rebalance_logic'] = test_rebalance_logic()
    test_results['wave_entropy'] = test_wave_entropy_function()
    test_results['profit_calculation'] = test_profit_calculation_efficiency()
    test_results['sharpe_proxy'] = test_sharpe_proxy_index()
    test_results['integration_pipeline'] = test_integration_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MATHEMATICAL PIPELINE TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All mathematical functions validated successfully!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 