#!/usr/bin/env python3
"""
Complete Mathematical Integration Test - Schwabot UROS v1.0
=========================================================

Tests all mathematical modules together in a complete pipeline:
- Bit Phase Engine
- Tensor Router
- Phase Entropy Matcher
- Matrix Mapper (updated)
- Profit Cycle Allocator (updated)
- GPU Offload Manager
- Hash Registry

This test validates the complete mathematical pipeline from hash input to profit allocation.
"""

import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_bit_phase_engine():
    """Test Bit Phase Engine functionality."""
    print("\n🧮 Testing Bit Phase Engine...")
    
    try:
        from core.bit_phase_engine import BitPhaseEngine
        
        engine = BitPhaseEngine()
        
        # Test hash
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        
        # Test different modes
        print(f"Testing hash: {test_hash[:16]}...")
        
        for mode in engine.supported_modes:
            phase = engine.resolve_bit_phase(test_hash, mode)
            print(f"  {mode}: {phase}")
        
        # Test optimal phase selection
        market_conditions = {
            'volatility': 0.15,
            'entropy_level': 5.2,
            'complexity': 0.7
        }
        
        optimal_phase, optimal_mode = engine.get_optimal_phase(test_hash, market_conditions)
        print(f"  Optimal phase: {optimal_phase} (mode: {optimal_mode})")
        
        # Test pattern analysis
        hash_sequence = [test_hash] * 10
        analysis = engine.analyze_phase_patterns(hash_sequence)
        print(f"  Pattern analysis: {len(analysis.get('phase_statistics', {}))} modes analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ Bit Phase Engine test failed: {e}")
        return False

def test_tensor_router():
    """Test Tensor Router functionality."""
    print("\n🔄 Testing Tensor Router...")
    
    try:
        from core.tensor_router import TensorRouter
        
        router = TensorRouter()
        
        # Test tensor score calculation
        entry_price = 100.0
        current_price = 110.0
        phase = 8
        
        tensor_score = router.tensor_score(entry_price, current_price, phase)
        print(f"  Tensor score: {tensor_score}")
        
        # Test trade routing
        market_conditions = {
            'volatility': 0.15,
            'entropy_level': 5.2,
            'complexity': 0.7
        }
        
        route = router.route_trade(entry_price, current_price, phase, market_conditions)
        print(f"  Route type: {route.route_type}")
        print(f"  Confidence: {route.confidence:.2f}")
        print(f"  Profit vector: {route.profit_vector}")
        
        # Test pattern analysis
        tensor_sequence = [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5]
        analysis = router.analyze_tensor_patterns(tensor_sequence)
        print(f"  Pattern analysis: {len(analysis.get('pattern_detection', {}).get('patterns', []))} patterns detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor Router test failed: {e}")
        return False

def test_phase_entropy_matcher():
    """Test Phase Entropy Matcher functionality."""
    print("\n🔗 Testing Phase Entropy Matcher...")
    
    try:
        from core.phase_entropy_matcher import PhaseEntropyMatcher
        
        matcher = PhaseEntropyMatcher()
        
        # Test phase weight matrix
        bit_pattern = [1, 0, 1, 1]
        entropy = 2.0
        
        phase_weight = matcher.phase_weight_matrix(bit_pattern, entropy)
        print(f"  Phase weight: {phase_weight}")
        
        # Test phase-entropy matching
        basket_id = "basket_0071"
        market_conditions = {
            'volatility': 0.15,
            'entropy_level': 5.2,
            'complexity': 0.7
        }
        
        match = matcher.match_phase_entropy(bit_pattern, entropy, basket_id, market_conditions)
        print(f"  Priority score: {match.priority_score:.4f}")
        print(f"  Basket ID: {match.basket_id}")
        
        # Test entropy pattern analysis
        entropy_sequence = [2.1, 3.5, 4.2, 5.8, 4.1, 3.9, 6.2, 5.1]
        analysis = matcher.analyze_entropy_patterns(entropy_sequence)
        print(f"  Entropy analysis: {len(analysis.get('pattern_detection', {}).get('patterns', []))} patterns detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase Entropy Matcher test failed: {e}")
        return False

def test_matrix_mapper_updated():
    """Test updated Matrix Mapper functionality."""
    print("\n📊 Testing Updated Matrix Mapper...")
    
    try:
        from core.matrix_mapper import MatrixMapper
        
        mapper = MatrixMapper()
        
        # Test match_basket_from_hash function
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        basket_id = mapper.match_basket_from_hash(test_hash)
        print(f"  Matched basket ID: {basket_id}")
        
        # Test decode_hash_to_basket function
        basket_result = mapper.decode_hash_to_basket(test_hash, 100, 45000.0)
        print(f"  Decoded basket: {basket_result}")
        
        # Test tensor score calculation
        tensor_score = mapper.calculate_tensor_score(44000.0, 45000.0, 8)
        print(f"  Matrix tensor score: {tensor_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix Mapper test failed: {e}")
        return False

def test_profit_cycle_allocator_updated():
    """Test updated Profit Cycle Allocator functionality."""
    print("\n💰 Testing Updated Profit Cycle Allocator...")
    
    try:
        from core.profit_cycle_allocator import ProfitCycleAllocator
        
        allocator = ProfitCycleAllocator()
        
        # Test rebalance function
        profit = 0.15
        volatility = 0.25
        
        rebalance_result = allocator.rebalance(profit, volatility)
        print(f"  Rebalance result: {rebalance_result}")
        
        # Test allocation function
        execution_packet = {
            'volume': 1000.0,
            'actual_profit': 500.0,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'tick': int(time.time())
        }
        
        market_data = {
            'price': 50000.0, 'volatility': 0.05, 'entropy_level': 4.2, 'complexity': 0.6,
            'trend_strength': 0.3, 'entry_exit_range': 0.02, 'liquidity_depth': 0.8,
            'trend_change_rate': 0.01, 'market_heat': 0.4, 'capital_exposure': 10000.0
        }
        
        allocation_result = allocator.allocate(
            execution_packet=execution_packet,
            cycles=['cycle1', 'cycle2', 'cycle3'],
            market_data=market_data
        )
        
        print(f"  Allocation success: {allocation_result.success}")
        print(f"  Tensor score: {allocation_result.tensor_score}")
        print(f"  Bit phase: {allocation_result.bit_phase}")
        
        return True
        
    except Exception as e:
        print(f"❌ Profit Cycle Allocator test failed: {e}")
        return False

def test_gpu_offload_manager():
    """Test GPU Offload Manager functionality."""
    print("\n🚀 Testing GPU Offload Manager...")
    
    try:
        from core.gpu_offload_manager import GPUOffloadManager
        
        manager = GPUOffloadManager()
        
        # Test bit phase resolution
        hash_strings = ["a1b2c3d4e5f6", "7890abcdef12", "345678901234"] * 100
        phases = manager.resolve_bit_phase_gpu(hash_strings, "8bit")
        print(f"  Resolved {len(phases)} bit phases")
        
        # Test tensor score calculation
        entry_prices = [100.0] * 300
        current_prices = [110.0] * 300
        phases = [8] * 300
        tensor_scores = manager.tensor_score_gpu(entry_prices, current_prices, phases)
        print(f"  Calculated {len(tensor_scores)} tensor scores")
        
        # Test wave entropy calculation
        sequences = [[1.0, 0.0, 1.0, 0.0]] * 300
        entropies = manager.wave_entropy_gpu(sequences)
        print(f"  Calculated {len(entropies)} entropy values")
        
        # Get performance metrics
        performance = manager.get_performance_metrics()
        print(f"  GPU Performance:")
        print(f"    Total operations: {performance.total_operations}")
        print(f"    Successful operations: {performance.successful_operations}")
        print(f"    Average execution time: {performance.average_execution_time_ms:.2f}ms")
        print(f"    GPU utilization: {performance.gpu_utilization:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU Offload Manager test failed: {e}")
        return False

def test_hash_registry():
    """Test Hash Registry functionality."""
    print("\n🗄️ Testing Hash Registry...")
    
    try:
        # Load hash registry
        with open("config/hash_registry.json", 'r') as f:
            registry = json.load(f)
        
        print(f"  Registry version: {registry['metadata']['version']}")
        print(f"  Total entries: {registry['metadata']['total_entries']}")
        print(f"  Basket count: {len(registry['baskets'])}")
        
        # Test specific basket
        basket_0071 = registry['baskets']['0071']
        print(f"  Basket 0071:")
        print(f"    Bit phase: {basket_0071['bit_phase']}")
        print(f"    Tensor score: {basket_0071['tensor_score']}")
        print(f"    Strategy: {basket_0071['strategy']}")
        
        # Test configuration
        config = registry['configuration']
        print(f"  Configuration:")
        print(f"    Max entries: {config['max_entries']}")
        print(f"    Hash length: {config['hash_length']}")
        print(f"    Basket ID range: {config['basket_id_range']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hash Registry test failed: {e}")
        return False

def test_complete_pipeline():
    """Test complete mathematical pipeline integration."""
    print("\n🔄 Testing Complete Mathematical Pipeline...")
    
    try:
        from core.bit_phase_engine import BitPhaseEngine
        from core.tensor_router import TensorRouter
        from core.phase_entropy_matcher import PhaseEntropyMatcher
        from core.matrix_mapper import MatrixMapper
        from core.profit_cycle_allocator import ProfitCycleAllocator
        from core.gpu_offload_manager import GPUOffloadManager
        
        # Initialize all components
        bit_engine = BitPhaseEngine()
        tensor_router = TensorRouter()
        entropy_matcher = PhaseEntropyMatcher()
        matrix_mapper = MatrixMapper()
        profit_allocator = ProfitCycleAllocator()
        gpu_manager = GPUOffloadManager()
        
        # Test complete pipeline
        print("  Running complete pipeline test...")
        
        # 1. Generate hash
        test_data = f"BTC_{int(time.time())}_{np.random.random()}"
        hash_value = hashlib.sha256(test_data.encode()).hexdigest()
        print(f"    Generated hash: {hash_value[:16]}...")
        
        # 2. Resolve bit phase
        market_conditions = {
            'volatility': 0.15,
            'entropy_level': 5.2,
            'complexity': 0.7
        }
        
        optimal_phase, optimal_mode = bit_engine.get_optimal_phase(hash_value, market_conditions)
        print(f"    Optimal phase: {optimal_phase} (mode: {optimal_mode})")
        
        # 3. Match basket
        basket_id = matrix_mapper.match_basket_from_hash(hash_value)
        print(f"    Matched basket ID: {basket_id}")
        
        # 4. Calculate tensor score
        entry_price = 50000.0
        current_price = 51000.0
        tensor_score = tensor_router.tensor_score(entry_price, current_price, optimal_phase)
        print(f"    Tensor score: {tensor_score}")
        
        # 5. Route trade
        route = tensor_router.route_trade(entry_price, current_price, optimal_phase, market_conditions)
        print(f"    Route type: {route.route_type}")
        print(f"    Profit vector: {route.profit_vector}")
        
        # 6. Phase-entropy matching
        bit_pattern = [1, 0, 1, 1, 0, 1, 0, 0]
        entropy = 4.2
        match = entropy_matcher.match_phase_entropy(bit_pattern, entropy, f"basket_{basket_id:04d}", market_conditions)
        print(f"    Priority score: {match.priority_score:.4f}")
        
        # 7. Profit allocation
        execution_packet = {
            'volume': 1000.0,
            'actual_profit': 500.0,
            'entry_price': entry_price,
            'current_price': current_price,
            'tick': int(time.time())
        }
        
        allocation_result = profit_allocator.allocate(
            execution_packet=execution_packet,
            market_data=market_data
        )
        
        print(f"    Allocation success: {allocation_result.success}")
        print(f"    Final tensor score: {allocation_result.tensor_score}")
        print(f"    Final bit phase: {allocation_result.bit_phase}")
        
        # 8. GPU acceleration (if available)
        if gpu_manager.gpu_available:
            gpu_phases = gpu_manager.resolve_bit_phase_gpu([hash_value], optimal_mode)
            gpu_tensor_scores = gpu_manager.tensor_score_gpu([entry_price], [current_price], [optimal_phase])
            print(f"    GPU acceleration: phases={gpu_phases}, scores={gpu_tensor_scores}")
        
        print("  ✅ Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 SCHWABOT UROS v1.0 - COMPLETE MATHEMATICAL INTEGRATION TEST")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        ("Bit Phase Engine", test_bit_phase_engine),
        ("Tensor Router", test_tensor_router),
        ("Phase Entropy Matcher", test_phase_entropy_matcher),
        ("Matrix Mapper (Updated)", test_matrix_mapper_updated),
        ("Profit Cycle Allocator (Updated)", test_profit_cycle_allocator_updated),
        ("GPU Offload Manager", test_gpu_offload_manager),
        ("Hash Registry", test_hash_registry),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = {}
    total_tests = len(tests)
    successful_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = {'success': success, 'error': None}
            if success:
                successful_tests += 1
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            results[test_name] = {'success': False, 'error': str(e)}
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Generate summary
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
    
    print(f"\n{'='*70}")
    print("📊 COMPLETE MATHEMATICAL INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.2%}")
    
    if success_rate >= 0.9:
        overall_status = "PASS"
        print(f"Overall Status: {overall_status} 🎉")
    elif success_rate >= 0.7:
        overall_status = "WARN"
        print(f"Overall Status: {overall_status} ⚠️")
    else:
        overall_status = "FAIL"
        print(f"Overall Status: {overall_status} ❌")
    
    # Export results
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'test_results': results,
            'mathematical_modules_tested': [
                'Bit Phase Engine',
                'Tensor Router', 
                'Phase Entropy Matcher',
                'Matrix Mapper (Updated)',
                'Profit Cycle Allocator (Updated)',
                'GPU Offload Manager',
                'Hash Registry',
                'Complete Pipeline Integration'
            ]
        }
        
        with open("complete_mathematical_integration_results.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Results exported to complete_mathematical_integration_results.json")
        
    except Exception as e:
        print(f"\n❌ Error exporting results: {e}")
    
    # Return exit code
    if overall_status == "PASS":
        print("\n🎉 All mathematical modules integrated successfully!")
        return 0
    elif overall_status == "WARN":
        print("\n⚠️ Some modules had issues. Review results.")
        return 1
    else:
        print("\n❌ Multiple modules failed. System needs attention.")
        return 2

if __name__ == "__main__":
    exit(main()) 