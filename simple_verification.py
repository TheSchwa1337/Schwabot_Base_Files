#!/usr/bin/env python3
"""
Simple Refactor Verification
===========================

Basic tests to verify the refactor resolved critical issues.
"""

def test_imports():
    """Test basic imports"""
    print("Testing imports...")
    
    try:
        import core
        print("✅ Core package imported")
    except Exception as e:
        print(f"❌ Core package import failed: {e}")
        return False
    
    try:
        from core.constants import WindowsCliCompatibilityHandler
        print("✅ WindowsCliCompatibilityHandler imported")
    except Exception as e:
        print(f"❌ WindowsCliCompatibilityHandler import failed: {e}")
        return False
    
    try:
        from core.fault_bus import FaultBus
        print("✅ FaultBus imported")
    except Exception as e:
        print(f"❌ FaultBus import failed: {e}")
        return False
    
    try:
        from tests.hooks.metrics import SchwabotMetrics
        print("✅ SchwabotMetrics imported")
    except Exception as e:
        print(f"❌ SchwabotMetrics import failed: {e}")
        return False
    
    return True

def test_instantiation():
    """Test class instantiation"""
    print("\nTesting instantiation...")
    
    try:
        from core.constants import WindowsCliCompatibilityHandler
        handler = WindowsCliCompatibilityHandler()
        print("✅ WindowsCliCompatibilityHandler instantiated")
    except Exception as e:
        print(f"❌ WindowsCliCompatibilityHandler instantiation failed: {e}")
        return False
    
    try:
        from core.fault_bus import FaultBus
        fault_bus = FaultBus()
        print("✅ FaultBus instantiated")
    except Exception as e:
        print(f"❌ FaultBus instantiation failed: {e}")
        return False
    
    try:
        from tests.hooks.metrics import SchwabotMetrics
        metrics = SchwabotMetrics()
        print("✅ SchwabotMetrics instantiated")
    except Exception as e:
        print(f"❌ SchwabotMetrics instantiation failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from core.constants import WindowsCliCompatibilityHandler
        handler = WindowsCliCompatibilityHandler()
        is_windows = handler.is_windows_cli()
        safe_msg = handler.safe_print("Test message")
        print(f"✅ Windows CLI compatibility working (is_windows: {is_windows})")
    except Exception as e:
        print(f"❌ Windows CLI compatibility failed: {e}")
        return False
    
    try:
        from core.fault_bus import FaultBus, FaultBusEvent, FaultType
        fault_bus = FaultBus()
        event = FaultBusEvent(
            tick=1,
            module="test",
            type=FaultType.THERMAL_HIGH,
            severity=0.5
        )
        fault_bus.push(event)
        print("✅ FaultBus operations working")
    except Exception as e:
        print(f"❌ FaultBus operations failed: {e}")
        return False
    
    try:
        from tests.hooks.metrics import SchwabotMetrics
        metrics = SchwabotMetrics()
        metrics.record_zygot_metric('drift_resonance', 0.75)
        print("✅ Metrics recording working")
    except Exception as e:
        print(f"❌ Metrics recording failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("SCHWABOT REFACTOR VERIFICATION")
    print("=" * 50)
    
    # Run tests
    import_ok = test_imports()
    instantiation_ok = test_instantiation()
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print(f"Imports: {'✅ PASS' if import_ok else '❌ FAIL'}")
    print(f"Instantiation: {'✅ PASS' if instantiation_ok else '❌ FAIL'}")
    print(f"Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if import_ok and instantiation_ok and functionality_ok:
        print("\n🎉 ALL TESTS PASSED! Refactor successful!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 