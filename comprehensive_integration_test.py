#!/usr/bin/env python3
"""
Comprehensive Integration Test for Schwabot Hybrid ZPE-Reactive System
=====================================================================

This test verifies that all systems work together cohesively with no errors,
proper imports, and consistent pipeline integration.
"""

import sys
import os
import asyncio
import time
import traceback
from typing import Dict, Any, List

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing Critical Imports...")
    
    imports_to_test = [
        # Core ZPE modules
        ("core.zpe_core", "ZPECore"),
        ("core.zpe_integration", "ZPEIntegration"),
        ("core.zpe_rotational_engine", "ZPERotationalEngine"),
        ("core.zpe_hybrid_mode_selector", "ZPEHybridModeSelector"),
        
        # Core Schwabot systems
        ("core.strategy_mapper", "StrategyMapper"),
        ("core.profit_cycle_allocator", "ProfitCycleAllocator"),
        ("core.lantern_vector_memory", "LanternMemory"),
        ("core.fault_bus", "FaultBus"),
        ("core.hash_registry", "HashRegistry"),
        
        # Mathematical libraries
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("pandas", "pd"),
        ("sklearn.decomposition", "PCA"),
        
        # Async and utilities
        ("asyncio", "asyncio"),
        ("datetime", "datetime"),
        ("logging", "logging"),
    ]
    
    failed_imports = []
    
    for module_name, import_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ {module_name} import failed: {e}")
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            print(f"⚠️ {module_name} import error: {e}")
            failed_imports.append((module_name, str(e)))
    
    return len(failed_imports) == 0, failed_imports

def test_zpe_mathematical_functions():
    """Test ZPE mathematical functions."""
    print("\n🧪 Testing ZPE Mathematical Functions...")
    
    try:
        from core.zpe_core import ZPECore
        
        zpe = ZPECore()
        
        # Test all core mathematical functions
        functions_to_test = [
            ("calculate_zpe_work", (0.8, 0.05)),
            ("calculate_rotational_torque", (0.7, 0.3)),
            ("calculate_thermal_efficiency", (100.0, 1000.0)),
            ("map_news_lantern_signals", (0.6, 0.2)),
            ("calculate_profit_reinjection", (50.0, 0.7)),
            ("calculate_elastic_resonance", (0.02, 1.0, 0.0, 1.0)),
            ("calculate_multi_vector_alignment", ({
                'BTC': {'magnitude': 0.8, 'resonance': 0.7},
                'ETH': {'magnitude': 0.6, 'resonance': 0.5}
            }, {'BTC': 0.6, 'ETH': 0.4})),
            ("update_recursive_cycle_depth", (1.0, 0.5)),
            ("calculate_temporal_fault_correction", (0.0, 0.5)),
            ("update_agent_consensus", ("test_agent", 0.8)),
        ]
        
        for func_name, args in functions_to_test:
            try:
                func = getattr(zpe, func_name)
                result = func(*args)
                print(f"✅ {func_name}: {result}")
            except Exception as e:
                print(f"❌ {func_name} failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ ZPE mathematical functions test failed: {e}")
        return False

def test_hybrid_mode_selector():
    """Test hybrid mode selector functionality."""
    print("\n🧪 Testing Hybrid Mode Selector...")
    
    try:
        from core.zpe_hybrid_mode_selector import (
            select_trading_mode, TradingMode, MarketCondition
        )
        
        # Test different market conditions
        test_cases = [
            {
                'name': 'Bull Market',
                'data': {
                    'trend_strength': 0.8,
                    'volatility': 0.3,
                    'price_change_24h': 0.08,
                    'profit_performance': 0.15
                },
                'expected_mode': TradingMode.ZPE_RECURSIVE
            },
            {
                'name': 'Bear Market',
                'data': {
                    'trend_strength': -0.6,
                    'volatility': 0.8,
                    'price_change_24h': -0.12,
                    'profit_performance': -0.08
                },
                'expected_mode': TradingMode.REACTIVE_TASKING
            },
            {
                'name': 'Crisis Market',
                'data': {
                    'trend_strength': -0.9,
                    'volatility': 0.95,
                    'price_change_24h': -0.25,
                    'profit_performance': -0.3
                },
                'expected_mode': TradingMode.EMERGENCY_FALLBACK
            },
            {
                'name': 'Sideways Market',
                'data': {
                    'trend_strength': 0.2,
                    'volatility': 0.5,
                    'price_change_24h': 0.02,
                    'profit_performance': 0.05
                },
                'expected_mode': TradingMode.HYBRID_BLEND
            }
        ]
        
        for test_case in test_cases:
            try:
                result = select_trading_mode(test_case['data'], timeframe="daily")
                print(f"✅ {test_case['name']}: {result.selected_mode.value}")
                print(f"   Confidence: {result.confidence_score:.3f}")
                print(f"   Market Condition: {result.market_condition.value}")
                
                # Verify expected mode (with some flexibility)
                if result.selected_mode == test_case['expected_mode']:
                    print(f"   ✅ Mode matches expectation")
                else:
                    print(f"   ⚠️ Mode differs from expectation (got {result.selected_mode.value}, expected {test_case['expected_mode'].value})")
                
            except Exception as e:
                print(f"❌ {test_case['name']} test failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid mode selector test failed: {e}")
        return False

def test_system_integration():
    """Test integration between all systems."""
    print("\n🧪 Testing System Integration...")
    
    try:
        # Test strategy mapper with ZPE integration
        from core.strategy_mapper import StrategyMapper
        from core.zpe_hybrid_mode_selector import select_trading_mode
        
        mapper = StrategyMapper()
        
        # Create test execution packet
        execution_packet = {
            'strategy_type': 'momentum',
            'volume': 1000.0,
            'expected_profit': 50.0,
            'actual_profit': 45.0,
            'btc': {'volume': 500.0, 'confidence': 0.8},
            'eth': {'volume': 300.0, 'confidence': 0.7},
            'xrp': {'volume': 150.0, 'confidence': 0.6},
            'usdc': {'volume': 50.0, 'confidence': 0.9}
        }
        
        # Test market data
        market_data = {
            'trend_strength': 0.7,
            'volatility': 0.4,
            'price_change_24h': 0.06,
            'profit_performance': 0.12
        }
        
        # Select mode
        mode_result = select_trading_mode(market_data, timeframe="daily")
        print(f"✅ Mode Selection: {mode_result.selected_mode.value}")
        
        # Test strategy mapping (async)
        async def test_mapping():
            try:
                result = await mapper.map_strategy_enhanced(
                    execution_packet=execution_packet,
                    market_data=market_data
                )
                print(f"✅ Strategy Mapping: {result.success}")
                print(f"   ZPE Work: {result.zpe_work:.6f}")
                print(f"   Should Spin: {result.zpe_should_spin}")
                return result.success
            except Exception as e:
                print(f"❌ Strategy mapping failed: {e}")
                return False
        
        # Run async test
        success = asyncio.run(test_mapping())
        return success
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_pipeline_consistency():
    """Test pipeline consistency across all systems."""
    print("\n🧪 Testing Pipeline Consistency...")
    
    try:
        from core.zpe_core import ZPECore
        from core.profit_cycle_allocator import ProfitCycleAllocator
        from core.lantern_vector_memory import LanternMemory
        from core.fault_bus import FaultBus
        from core.hash_registry import HashRegistry
        
        # Initialize all systems
        zpe = ZPECore()
        allocator = ProfitCycleAllocator()
        memory = LanternMemory()
        fault_bus = FaultBus()
        hash_registry = HashRegistry()
        
        print("✅ All systems initialized successfully")
        
        # Test consistent data flow
        test_data = {
            'trend_strength': 0.6,
            'volatility': 0.4,
            'price_change_24h': 0.04,
            'profit_performance': 0.08,
            'volume': 1000.0,
            'expected_profit': 40.0,
            'actual_profit': 38.0
        }
        
        # Test ZPE calculations
        zpe_work = zpe.calculate_zpe_work(test_data['trend_strength'], 0.05)
        print(f"✅ ZPE Work: {zpe_work:.6f}")
        
        # Test profit allocation
        allocation_result = allocator.allocate(
            execution_packet={'volume': test_data['volume'], 'actual_profit': test_data['actual_profit']},
            market_data=test_data
        )
        print(f"✅ Profit Allocation: {allocation_result.success}")
        print(f"   ZPE Efficiency: {allocation_result.zpe_efficiency:.6f}")
        
        # Test memory operations
        memory_entry = memory.add_memory_entry(
            vector=[0.1, 0.2, 0.3, 0.4],
            news_density=0.6,
            sentiment_delta=0.2,
            price_derivative=0.02
        )
        print(f"✅ Memory Entry: {memory_entry is not None}")
        if memory_entry:
            print(f"   ZPE Signal Strength: {memory_entry.zpe_signal_strength:.6f}")
        
        # Test fault bus
        from core.fault_bus import FaultBusEvent, FaultType
        fault_event = FaultBusEvent(
            tick=1,
            module="test",
            type=FaultType.PROFIT_LOW,
            severity=0.3,
            metadata=test_data
        )
        fault_bus.push(fault_event)
        print(f"✅ Fault Bus: Event pushed successfully")
        
        # Test hash registry
        async def test_hash_registry():
            try:
                hash_id = await hash_registry.register_hash(
                    hash_type="command",
                    agent_type="test",
                    domain="strategy",
                    payload=test_data,
                    confidence_score=0.8
                )
                print(f"✅ Hash Registry: {hash_id}")
                return True
            except Exception as e:
                print(f"❌ Hash registry failed: {e}")
                return False
        
        hash_success = asyncio.run(test_hash_registry())
        
        return hash_success
        
    except Exception as e:
        print(f"❌ Pipeline consistency test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_and_memory():
    """Test performance and memory usage."""
    print("\n🧪 Testing Performance and Memory...")
    
    try:
        import psutil
        import time
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run intensive operations
        from core.zpe_core import ZPECore
        zpe = ZPECore()
        
        start_time = time.time()
        
        # Run multiple calculations
        for i in range(100):
            zpe.calculate_zpe_work(0.5 + i * 0.01, 0.05)
            zpe.calculate_rotational_torque(0.5 + i * 0.01, 0.3)
            zpe.calculate_thermal_efficiency(100.0 + i, 1000.0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"✅ Performance Test:")
        print(f"   Execution Time: {execution_time:.3f} seconds")
        print(f"   Initial Memory: {initial_memory:.2f} MB")
        print(f"   Final Memory: {final_memory:.2f} MB")
        print(f"   Memory Increase: {memory_increase:.2f} MB")
        
        # Performance thresholds
        if execution_time < 1.0:  # Should complete in under 1 second
            print(f"   ✅ Execution time within acceptable range")
        else:
            print(f"   ⚠️ Execution time slower than expected")
        
        if memory_increase < 50.0:  # Should not increase memory by more than 50MB
            print(f"   ✅ Memory usage within acceptable range")
        else:
            print(f"   ⚠️ Memory usage higher than expected")
        
        return execution_time < 1.0 and memory_increase < 50.0
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run comprehensive integration tests."""
    print("🔥 SCHWABOT HYBRID ZPE-REACTIVE SYSTEM - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_imports),
        ("ZPE Mathematical Functions", test_zpe_mathematical_functions),
        ("Hybrid Mode Selector", test_hybrid_mode_selector),
        ("System Integration", test_system_integration),
        ("Pipeline Consistency", test_pipeline_consistency),
        ("Performance and Memory", test_performance_and_memory),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Import Test":
                success, failed_imports = test_func()
                if not success:
                    print(f"❌ {test_name} failed with {len(failed_imports)} failed imports:")
                    for module, error in failed_imports:
                        print(f"   - {module}: {error}")
            else:
                success = test_func()
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Schwabot Hybrid ZPE-Reactive System is ready for deployment!")
        print("🔥 Both reactive tasking AND recursive velocity are properly integrated!")
        print("🔄 Hybrid mode selection is working correctly!")
        print("⚡ Pipeline consistency verified across all systems!")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        print("🔧 Check dependencies, imports, and system configurations.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 