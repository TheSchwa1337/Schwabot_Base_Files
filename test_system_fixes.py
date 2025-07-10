#!/usr/bin/env python3
"""
Test System Fixes - Verify Core Components
==========================================

This script tests the fixed core components to ensure they are working properly.
"""

import sys
import time
import traceback

def test_chrono_resonance_weather_mapper():
    """Test the chrono resonance weather mapper."""
    print("Testing Chrono Resonance Weather Mapper...")
    try:
        from core.chrono_resonance_weather_mapper import ChronoResonanceWeatherMapper
        
        mapper = ChronoResonanceWeatherMapper()
        result = mapper.compute_crwf(1.0, 40.0, -74.0, 100.0)
        
        print(f"✅ CRWF Result: {result}")
        print("✅ Chrono Resonance Weather Mapper working")
        return True
    except Exception as e:
        print(f"❌ Chrono Resonance Weather Mapper failed: {e}")
        traceback.print_exc()
        return False

def test_temporal_warp_engine():
    """Test the temporal warp engine."""
    print("\nTesting Temporal Warp Engine...")
    try:
        from core.temporal_warp_engine import TemporalWarpEngine
        
        engine = TemporalWarpEngine()
        projected_time = engine.calculate_temporal_projection(time.time(), 0.1)
        
        print(f"✅ Projected Time: {projected_time}")
        print("✅ Temporal Warp Engine working")
        return True
    except Exception as e:
        print(f"❌ Temporal Warp Engine failed: {e}")
        traceback.print_exc()
        return False

def test_cli_live_entry():
    """Test the CLI live entry system."""
    print("\nTesting CLI Live Entry System...")
    try:
        from core.cli_live_entry import SchwabotCLI
        
        cli = SchwabotCLI()
        status = cli.get_system_status()
        
        print(f"✅ CLI Status: {status['initialized']}")
        print("✅ CLI Live Entry System working")
        return True
    except Exception as e:
        print(f"❌ CLI Live Entry System failed: {e}")
        traceback.print_exc()
        return False

def test_clean_unified_math():
    """Test the clean unified math system."""
    print("\nTesting Clean Unified Math System...")
    try:
        from core.clean_unified_math import CleanUnifiedMathSystem
        
        math_system = CleanUnifiedMathSystem()
        result = math_system.optimize_profit(100.0, 0.5, 0.8)
        
        print(f"✅ Optimized Profit: {result}")
        print("✅ Clean Unified Math System working")
        return True
    except Exception as e:
        print(f"❌ Clean Unified Math System failed: {e}")
        traceback.print_exc()
        return False

def test_backend_math():
    """Test the backend math system."""
    print("\nTesting Backend Math System...")
    try:
        from core.backend_math import backend_info, get_backend
        
        info = backend_info()
        backend = get_backend()
        
        print(f"✅ Backend Info: {info}")
        print(f"✅ Backend Type: {type(backend).__name__}")
        print("✅ Backend Math System working")
        return True
    except Exception as e:
        print(f"❌ Backend Math System failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔮 Testing Schwabot Core Components")
    print("=" * 50)
    
    tests = [
        test_backend_math,
        test_clean_unified_math,
        test_chrono_resonance_weather_mapper,
        test_temporal_warp_engine,
        test_cli_live_entry,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. System needs additional fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 