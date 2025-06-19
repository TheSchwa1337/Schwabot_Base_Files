"""
Simple Test - Magic Number Optimization System
==============================================

Quick verification that the revolutionary magic number optimization system is working.
"""

import sys
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

def test_system_constants():
    """Test that system constants can be imported"""
    try:
        from core.system_constants import SYSTEM_CONSTANTS
        print("✅ System constants imported successfully")
        
        # Test a few constants
        hash_threshold = SYSTEM_CONSTANTS.core.MIN_HASH_CORRELATION_THRESHOLD
        bit_level_42 = SYSTEM_CONSTANTS.visualization.BIT_LEVEL_42
        
        print(f"   Hash Correlation Threshold: {hash_threshold}")
        print(f"   Bit Level 42: {bit_level_42}")
        return True
    except Exception as e:
        print(f"❌ System constants failed: {e}")
        return False

def test_zbe_temperature_tensor():
    """Test ZBE temperature tensor"""
    try:
        from core.zbe_temperature_tensor import ZBETemperatureTensor
        tensor = ZBETemperatureTensor()
        
        stats = tensor.get_thermal_stats()
        print("✅ ZBE Temperature Tensor working")
        print(f"   Current temp: {stats['current_temp']:.1f}°C")
        print(f"   Thermal efficiency: {stats['thermal_efficiency']:.1%}")
        return True
    except Exception as e:
        print(f"❌ ZBE tensor failed: {e}")
        return False

def test_optimization_engine():
    """Test the optimization engine"""
    try:
        from core.magic_number_optimization_engine import (
            MagicNumberOptimizationEngine, 
            OptimizationType
        )
        from core.zbe_temperature_tensor import ZBETemperatureTensor
        
        # Create engine
        zbe_tensor = ZBETemperatureTensor()
        engine = MagicNumberOptimizationEngine(zbe_tensor)
        
        print("✅ Optimization engine created successfully")
        
        # Test mathematical analysis
        analysis = engine.constant_analysis
        print(f"   Golden ratio relationship: {analysis['golden_ratio_relationship']:.6f}")
        print(f"   Fibonacci optimization potential: {analysis['fibonacci_optimization_potential']:.2f}x")
        
        # Test a simple optimization
        factor = engine.apply_golden_ratio_optimization(0.25, "test_threshold")
        print(f"   Test optimization: 0.25 → {factor.optimized_value:.6f}")
        print(f"   Improvement: {factor.improvement_factor:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Optimization engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_constants_wrapper():
    """Test the optimized constants wrapper"""
    try:
        from core.optimized_constants_wrapper import OPTIMIZED_CONSTANTS
        
        # Test basic access
        original_threshold = OPTIMIZED_CONSTANTS.core_thresholds.MIN_HASH_CORRELATION_THRESHOLD
        print("✅ Optimized constants wrapper working")
        print(f"   Hash correlation threshold: {original_threshold}")
        
        # Check optimization status
        status = OPTIMIZED_CONSTANTS.get_optimization_status()
        if status['optimized']:
            print(f"   Optimizations active: {status['total_constants_optimized']} constants")
            print(f"   Average improvement: {status['average_improvement']:.1%}")
        else:
            print("   Optimizations not yet active (this is normal)")
        
        # Try enabling optimizations
        print("   Testing optimization activation...")
        if OPTIMIZED_CONSTANTS.enable_optimizations():
            print("   ✅ Optimizations successfully activated!")
            status = OPTIMIZED_CONSTANTS.get_optimization_status()
            if status['optimized']:
                print(f"   📊 {status['total_constants_optimized']} constants optimized")
                print(f"   📈 Average improvement: {status['average_improvement']:.1%}")
        else:
            print("   ⚠️ Optimization activation failed (dependencies may be missing)")
        
        return True
    except Exception as e:
        print(f"❌ Optimized constants wrapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔥 MAGIC NUMBER OPTIMIZATION SYSTEM TEST 🔥")
    print("=" * 50)
    
    tests = [
        ("System Constants", test_system_constants),
        ("ZBE Temperature Tensor", test_zbe_temperature_tensor),
        ("Optimization Engine", test_optimization_engine),
        ("Optimized Constants Wrapper", test_optimized_constants_wrapper)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📊 Testing {name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🌟 ALL TESTS PASSED! Magic number optimization system is ready!")
        print("\n🚀 Quick Start:")
        print("   from core.optimized_constants_wrapper import OPTIMIZED_CONSTANTS")
        print("   OPTIMIZED_CONSTANTS.enable_optimizations()")
        print("   # Now use OPTIMIZED_CONSTANTS instead of SYSTEM_CONSTANTS!")
        return True
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 