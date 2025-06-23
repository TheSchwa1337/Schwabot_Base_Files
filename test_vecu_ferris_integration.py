#!/usr/bin/env python3
"""
Comprehensive Test Script for VECU and Ferris RDE Integration
============================================================

This script tests the complete integration between:
- VECU (Vectorized Electronic Control Unit)
- Ferris RDE (Recursive Dynamic Engine)
- Unified Mathematics System
- All existing Schwabot systems

Ensures everything works together cohesively with proper error handling.
"""

import sys
import os
import asyncio
import time
import traceback
from typing import Dict, Any, List
from datetime import datetime

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing Critical Imports...")
    
    imports_to_test = [
        # Core VECU and Ferris RDE modules
        ("core.vecu_core", "VECUCore"),
        ("core.ferris_rde_core", "FerrisRDECore"),
        ("core.unified_mathematics_config", "get_unified_math"),
        
        # Existing Schwabot systems
        ("core.strategy_mapper", "StrategyMapper"),
        ("core.profit_cycle_allocator", "ProfitCycleAllocator"),
        ("core.lantern_vector_memory", "LanternVectorMemory"),
        ("core.fault_bus", "FaultBus"),
        ("core.hash_registry", "HashRegistry"),
        
        # ZPE framework
        ("core.zpe_core", "ZPECore"),
        ("core.zpe_integration", "ZPEIntegration"),
        ("core.zpe_rotational_engine", "ZPERotationalEngine"),
        ("core.zpe_hybrid_mode_selector", "ZPEHybridModeSelector"),
        
        # Live backtesting systems
        ("core.trajectory_sphere", "TrajectorySphere"),
        ("core.demo_memory_core", "DemoMemoryCore"),
    ]
    
    successful_imports = 0
    total_imports = len(imports_to_test)
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name} imported successfully")
                successful_imports += 1
            else:
                print(f"⚠️ {module_name} imported but {class_name} not found")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
        except Exception as e:
            print(f"❌ Error importing {module_name}: {e}")
    
    print(f"📊 Import Results: {successful_imports}/{total_imports} successful")
    return successful_imports == total_imports

def test_vecu_core():
    """Test VECU core functionality."""
    print("\n🧪 Testing VECU Core...")
    
    try:
        from core.vecu_core import get_vecu_core, vecu_timing_sync, pwm_profit_injection, vecu_feedback_loop
        
        vecu = get_vecu_core()
        
        # Test timing synchronization
        timing_data = vecu_timing_sync(
            tick_id=1,
            rpm_equivalent=0.8,
            entropy_level=0.6
        )
        print(f"✅ VECU Timing Sync: Amplification = {timing_data.profit_amplification:.6f}")
        
        # Test PWM injection
        injection_data = pwm_profit_injection(
            current_phase=0.5,
            profit_potential=100.0,
            market_volatility=0.3
        )
        print(f"✅ PWM Injection: Voltage = {injection_data.profit_voltage:.6f}")
        
        # Test feedback loop
        feedback_data = vecu_feedback_loop(
            predicted_profit=50.0,
            actual_profit=45.0,
            previous_phase=0.3,
            timing_data=timing_data
        )
        print(f"✅ VECU Feedback: Error = {feedback_data.error_delta:.6f}")
        
        # Get statistics
        stats = vecu.get_vecu_statistics()
        print(f"✅ VECU Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ VECU Core test failed: {e}")
        traceback.print_exc()
        return False

def test_ferris_rde_core():
    """Test Ferris RDE core functionality."""
    print("\n🧪 Testing Ferris RDE Core...")
    
    try:
        from core.ferris_rde_core import (
            get_ferris_rde_core, update_ferris_wheel, map_btc_price_16bit,
            create_matrix_basket, formulate_trade_walls, integrate_with_vecu
        )
        
        ferris = get_ferris_rde_core()
        
        # Test market data
        test_market_data = {
            'btc_price': 50000.0,
            'volume_btc': 1000.0,
            'volume_eth': 500.0,
            'volume_xrp': 100.0,
            'volume_usdc': 100.0,
            'volatility': 0.3,
            'trend_strength': 0.2
        }
        
        # Test Ferris wheel update
        wheel_data = update_ferris_wheel()
        print(f"✅ Ferris Wheel: Phase = {wheel_data.phase.value}, Height = {wheel_data.height:.3f}")
        
        # Test BTC price mapping
        price_data = map_btc_price_16bit(test_market_data['btc_price'])
        print(f"✅ Price Mapping: {price_data.btc_price:.2f} → {price_data.mapped_price} (16-bit)")
        
        # Test matrix basket creation
        basket_data = create_matrix_basket(test_market_data)
        print(f"✅ Matrix Basket: {basket_data.basket_id}, Resonance = {basket_data.resonance_score:.3f}")
        
        # Test trade wall formulation
        buy_wall, sell_wall = formulate_trade_walls(test_market_data, basket_data)
        print(f"✅ Trade Walls: Buy confidence = {buy_wall.confidence_score:.3f}, Sell confidence = {sell_wall.confidence_score:.3f}")
        
        # Test VECU integration
        integration_result = integrate_with_vecu(wheel_data, price_data, basket_data)
        if integration_result:
            print(f"✅ VECU Integration: Amplification = {integration_result.get('vecu_amplification', 0.0):.6f}")
        
        # Get statistics
        stats = ferris.get_ferris_statistics()
        print(f"✅ Ferris RDE Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ferris RDE Core test failed: {e}")
        traceback.print_exc()
        return False

def test_unified_mathematics():
    """Test unified mathematics system."""
    print("\n🧪 Testing Unified Mathematics System...")
    
    try:
        from core.unified_mathematics_config import get_unified_math
        
        unified_math = get_unified_math()
        
        # Test basic mathematical operations
        result = unified_math.calculate_zpe_work(volume=100.0, pressure=0.8, temperature=0.6)
        print(f"✅ ZPE Work Calculation: {result:.6f}")
        
        result = unified_math.calculate_thermal_efficiency(work_input=50.0, work_output=45.0)
        print(f"✅ Thermal Efficiency: {result:.6f}")
        
        result = unified_math.calculate_elastic_resonance(frequency=0.5, amplitude=0.8, damping=0.2)
        print(f"✅ Elastic Resonance: {result:.6f}")
        
        # Test performance monitoring
        performance = unified_math.get_performance_stats()
        print(f"✅ Performance Stats: {performance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified Mathematics test failed: {e}")
        traceback.print_exc()
        return False

def test_existing_systems():
    """Test existing Schwabot systems."""
    print("\n🧪 Testing Existing Schwabot Systems...")
    
    try:
        # Test Strategy Mapper
        from core.strategy_mapper import StrategyMapper
        strategy_mapper = StrategyMapper()
        print("✅ Strategy Mapper initialized")
        
        # Test Profit Cycle Allocator
        from core.profit_cycle_allocator import ProfitCycleAllocator
        profit_allocator = ProfitCycleAllocator()
        print("✅ Profit Cycle Allocator initialized")
        
        # Test Lantern Vector Memory
        from core.lantern_vector_memory import LanternVectorMemory
        lantern_memory = LanternVectorMemory()
        print("✅ Lantern Vector Memory initialized")
        
        # Test Fault Bus
        from core.fault_bus import FaultBus
        fault_bus = FaultBus()
        print("✅ Fault Bus initialized")
        
        # Test Hash Registry
        from core.hash_registry import HashRegistry
        hash_registry = HashRegistry()
        print("✅ Hash Registry initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing systems test failed: {e}")
        traceback.print_exc()
        return False

def test_zpe_framework():
    """Test ZPE mathematical framework."""
    print("\n🧪 Testing ZPE Mathematical Framework...")
    
    try:
        from core.zpe_core import ZPECore
        from core.zpe_integration import ZPEIntegration
        from core.zpe_rotational_engine import ZPERotationalEngine
        from core.zpe_hybrid_mode_selector import ZPEHybridModeSelector
        
        # Test ZPE Core
        zpe_core = ZPECore()
        print("✅ ZPE Core initialized")
        
        # Test ZPE Integration
        zpe_integration = ZPEIntegration()
        print("✅ ZPE Integration initialized")
        
        # Test ZPE Rotational Engine
        zpe_engine = ZPERotationalEngine()
        print("✅ ZPE Rotational Engine initialized")
        
        # Test Hybrid Mode Selector
        hybrid_selector = ZPEHybridModeSelector()
        print("✅ ZPE Hybrid Mode Selector initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ ZPE Framework test failed: {e}")
        traceback.print_exc()
        return False

def test_live_backtesting():
    """Test live backtesting systems."""
    print("\n🧪 Testing Live Backtesting Systems...")
    
    try:
        from core.trajectory_sphere import TrajectorySphere
        from core.demo_memory_core import DemoMemoryCore
        
        # Test Trajectory Sphere
        trajectory_sphere = TrajectorySphere()
        print("✅ Trajectory Sphere initialized")
        
        # Test Demo Memory Core
        demo_memory = DemoMemoryCore()
        print("✅ Demo Memory Core initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Live backtesting test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test complete integration workflow."""
    print("\n🧪 Testing Complete Integration Workflow...")
    
    try:
        # Import all necessary components
        from core.vecu_core import get_vecu_core
        from core.ferris_rde_core import get_ferris_rde_core
        from core.unified_mathematics_config import get_unified_math
        from core.strategy_mapper import StrategyMapper
        from core.profit_cycle_allocator import ProfitCycleAllocator
        
        # Initialize components
        vecu = get_vecu_core()
        ferris = get_ferris_rde_core()
        unified_math = get_unified_math()
        strategy_mapper = StrategyMapper()
        profit_allocator = ProfitCycleAllocator()
        
        # Simulate complete workflow
        print("🔄 Simulating complete integration workflow...")
        
        # 1. Update Ferris wheel
        wheel_data = ferris.update_ferris_wheel()
        
        # 2. Map BTC price
        price_data = ferris.map_btc_price_16bit(50000.0)
        
        # 3. Create matrix basket
        market_data = {'btc_price': 50000.0, 'volume_btc': 1000.0, 'volatility': 0.3}
        basket_data = ferris.create_matrix_basket(market_data)
        
        # 4. Get VECU timing
        timing_data = vecu.vecu_timing_sync(
            tick_id=1,
            rpm_equivalent=wheel_data.velocity * 60 / (2 * 3.14159),
            entropy_level=price_data.mapped_price / 65535.0
        )
        
        # 5. Get PWM injection
        injection_data = vecu.pwm_profit_injection(
            current_phase=wheel_data.height,
            profit_potential=basket_data.resonance_score * 100.0,
            market_volatility=0.3
        )
        
        # 6. Calculate unified mathematics
        zpe_work = unified_math.calculate_zpe_work(
            volume=injection_data.profit_voltage,
            pressure=timing_data.profit_amplification,
            temperature=basket_data.resonance_score
        )
        
        print(f"✅ Complete Workflow: ZPE Work = {zpe_work:.6f}")
        print(f"   Ferris Phase: {wheel_data.phase.value}")
        print(f"   Price Triggered: {price_data.is_triggered}")
        print(f"   VECU Amplification: {timing_data.profit_amplification:.6f}")
        print(f"   PWM Voltage: {injection_data.profit_voltage:.6f}")
        print(f"   Basket Resonance: {basket_data.resonance_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🔥 SCHWABOT VECU & FERRIS RDE INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Critical Imports", test_imports),
        ("VECU Core", test_vecu_core),
        ("Ferris RDE Core", test_ferris_rde_core),
        ("Unified Mathematics", test_unified_mathematics),
        ("Existing Systems", test_existing_systems),
        ("ZPE Framework", test_zpe_framework),
        ("Live Backtesting", test_live_backtesting),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            successful_tests += 1
    
    print(f"\n📈 Overall Results: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED! VECU and Ferris RDE integration is complete.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 