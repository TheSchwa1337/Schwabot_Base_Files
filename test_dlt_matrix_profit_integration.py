#!/usr/bin/env python3
"""
DLT Matrix Profit Integration Test - Schwabot UROS v1.0
=====================================================

Comprehensive integration test for the complete trading system pipeline.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components
try:
    from core.dlt_waveform_engine import DLTWaveformEngine, BitPhase as DLTBitPhase
    from core.matrix_mapper import MatrixMapper, BitPhase as MatrixBitPhase
    from core.profit_cycle_allocator import ProfitCycleAllocator, allocate_profit_cycle
    from core.zpe_core import ZPECore
    CORE_COMPONENTS_AVAILABLE = True
    print("✅ All core components imported successfully")
except ImportError as e:
    CORE_COMPONENTS_AVAILABLE = False
    print(f"❌ Some core components not available: {e}")


def test_component_initialization():
    """Test initialization of all components."""
    print("\n🧪 Testing Component Initialization...")
    
    results = {}
    
    try:
        # Test DLT Waveform Engine
        dlt_engine = DLTWaveformEngine()
        results['dlt_engine'] = {'status': 'success', 'gpu_available': dlt_engine.gpu_available}
        print(f"✅ DLT Waveform Engine: GPU={dlt_engine.gpu_available}")

        # Test Matrix Mapper
        matrix_mapper = MatrixMapper()
        results['matrix_mapper'] = {'status': 'success', 'hash_registry_size': len(matrix_mapper.hash_registry)}
        print(f"✅ Matrix Mapper: Hash Registry={len(matrix_mapper.hash_registry)}")

        # Test Profit Cycle Allocator
        profit_allocator = ProfitCycleAllocator()
        results['profit_allocator'] = {'status': 'success', 'strategy': profit_allocator.allocation_strategy}
        print(f"✅ Profit Cycle Allocator: Strategy={profit_allocator.allocation_strategy}")

        # Test ZPE Core
        zpe_core = ZPECore()
        results['zpe_core'] = {'status': 'success', 'recursion_depth': zpe_core.recursion_depth}
        print(f"✅ ZPE Core: Recursion Depth={zpe_core.recursion_depth}")

    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Component initialization failed: {e}")
    
    return results


def test_dlt_waveform_processing():
    """Test DLT waveform processing with quantum integration."""
    print("\n🌊 Testing DLT Waveform Processing...")
    
    results = {}
    
    try:
        dlt_engine = DLTWaveformEngine()
        
        # Generate test waveform data
        t = np.linspace(0, 10, 1000)
        waveform_data = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
        
        # Test waveform processing
        waveform_result = dlt_engine.process_waveform_data(
            name="test_waveform",
            x=waveform_data,
            sample_rate=1.0
        )
        
        results['waveform_processing'] = {
            'success': waveform_result.get('success', False),
            'tensor_score': waveform_result.get('tensor_score', 0.0)
        }
        print(f"✅ Waveform Processing: Success={waveform_result.get('success', False)}")

        # Test matrix basket creation
        market_data = {
            'price': 50000.0, 'volatility': 0.05, 'entropy_level': 4.2, 'complexity': 0.6,
            'assets': ['BTC', 'ETH', 'ADA', 'DOT', 'SOL']
        }
        basket = dlt_engine.create_matrix_basket(market_data)
        results['matrix_basket'] = {
            'basket_id': basket.basket_id,
            'bit_phase': basket.bit_phase.value,
            'resonance_score': basket.resonance_score
        }
        print(f"✅ Matrix Basket: ID={basket.basket_id}, Phase={basket.bit_phase.value}")

    except Exception as e:
        results['error'] = str(e)
        print(f"❌ DLT waveform processing failed: {e}")
    
    return results


def test_matrix_mapper_functionality():
    """Test matrix mapper functionality with hash-basket matching."""
    print("\n🔗 Testing Matrix Mapper Functionality...")
    
    results = {}
    
    try:
        matrix_mapper = MatrixMapper()
        
        # Test hash decoding
        test_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
        basket_id = matrix_mapper.decode_hash_to_basket(test_hash, 100, 50000.0)
        results['hash_decoding'] = {'basket_id': basket_id, 'status': 'success'}
        print(f"✅ Hash Decoding: Basket ID={basket_id}")

        # Test profit allocation
        market_data = {
            'price': 50000.0, 'volatility': 0.05, 'entropy_level': 4.2, 'complexity': 0.6
        }
        allocation = matrix_mapper.allocate_profit(1000.0, market_data)
        results['profit_allocation'] = {
            'allocation_id': allocation.allocation_id if allocation else None,
            'basket_id': allocation.basket_id if allocation else None,
            'tensor_score': allocation.tensor_score if allocation else 0.0
        }
        print(f"✅ Profit Allocation: ID={allocation.allocation_id if allocation else 'None'}")

    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Matrix mapper functionality failed: {e}")
    
    return results


def test_profit_cycle_allocation():
    """Test profit cycle allocation with tensor scoring."""
    print("\n💰 Testing Profit Cycle Allocation...")
    
    results = {}
    
    try:
        # Create test execution packet
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

        # Test allocation
        allocation_result = allocate_profit_cycle(
            execution_packet=execution_packet,
            cycles=['cycle1', 'cycle2', 'cycle3'],
            market_data=market_data
        )
        
        results['allocation'] = {
            'success': allocation_result.success,
            'tensor_score': allocation_result.tensor_score,
            'bit_phase': allocation_result.bit_phase,
            'zpe_efficiency': allocation_result.zpe_efficiency
        }
        print(f"✅ Allocation: Success={allocation_result.success}, Tensor Score={allocation_result.tensor_score:.4f}")

    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Profit cycle allocation failed: {e}")
    
    return results


def test_complete_pipeline_integration():
    """Test complete pipeline integration from waveform to profit allocation."""
    print("\n🔄 Testing Complete Pipeline Integration...")
    
    results = {}
    
    try:
        # Step 1: Process waveform data
        dlt_engine = DLTWaveformEngine()
        t = np.linspace(0, 10, 1000)
        waveform_data = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
        
        waveform_result = dlt_engine.process_waveform_data(
            name="pipeline_waveform",
            x=waveform_data,
            sample_rate=1.0
        )
        
        results['step1_waveform'] = {
            'success': waveform_result.get('success', False),
            'tensor_score': waveform_result.get('tensor_score', 0.0)
        }
        print(f"✅ Step 1 - Waveform Processing: Tensor Score={waveform_result.get('tensor_score', 0.0):.4f}")

        # Step 2: Create matrix basket
        market_data = {
            'price': 50000.0, 'volatility': 0.05, 'entropy_level': 4.2, 'complexity': 0.6,
            'assets': ['BTC', 'ETH', 'ADA', 'DOT', 'SOL']
        }
        basket = dlt_engine.create_matrix_basket(market_data)
        results['step2_matrix_basket'] = {
            'success': True,
            'basket_id': basket.basket_id,
            'resonance_score': basket.resonance_score
        }
        print(f"✅ Step 2 - Matrix Basket: ID={basket.basket_id}, Resonance={basket.resonance_score:.4f}")

        # Step 3: Allocate profit
        execution_packet = {
            'volume': 1000.0,
            'actual_profit': 500.0,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'tick': int(time.time())
        }
        
        allocation_result = allocate_profit_cycle(
            execution_packet=execution_packet,
            market_data=market_data
        )
        
        results['step3_profit_allocation'] = {
            'success': allocation_result.success,
            'tensor_score': allocation_result.tensor_score,
            'bit_phase': allocation_result.bit_phase
        }
        print(f"✅ Step 3 - Profit Allocation: Success={allocation_result.success}, Tensor Score={allocation_result.tensor_score:.4f}")

        # Verify pipeline success
        pipeline_success = (
            waveform_result.get('success', False) and
            basket.basket_id and
            allocation_result.success
        )
        
        results['pipeline_integration'] = {
            'success': pipeline_success,
            'all_steps_completed': True
        }
        print(f"✅ Pipeline Integration: Success={pipeline_success}")

    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Complete pipeline integration failed: {e}")
    
    return results


def main():
    """Main function to run the integration tests."""
    print("DLT Matrix Profit Integration Test - Schwabot UROS v1.0")
    print("=" * 60)
    
    if not CORE_COMPONENTS_AVAILABLE:
        print("❌ Core components not available - skipping tests")
        return 1
    
    # Run tests
    test_functions = [
        ('component_initialization', test_component_initialization),
        ('dlt_waveform_processing', test_dlt_waveform_processing),
        ('matrix_mapper_functionality', test_matrix_mapper_functionality),
        ('profit_cycle_allocation', test_profit_cycle_allocation),
        ('complete_pipeline_integration', test_complete_pipeline_integration)
    ]
    
    all_results = {}
    successful_tests = 0
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*20} {test_name.upper()} {'='*20}")
            result = test_func()
            all_results[test_name] = result
            
            if 'error' not in result:
                successful_tests += 1
                print(f"✅ {test_name} completed successfully")
            else:
                print(f"❌ {test_name} failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            all_results[test_name] = {'error': str(e)}
    
    # Summary
    total_tests = len(test_functions)
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Overall Status: {'PASS' if success_rate >= 0.8 else 'FAIL'}")
    
    # Save results
    try:
        with open("dlt_matrix_profit_integration_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("✅ Test results saved to dlt_matrix_profit_integration_results.json")
    except Exception as e:
        print(f"❌ Failed to save test results: {e}")
    
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    exit(main()) 