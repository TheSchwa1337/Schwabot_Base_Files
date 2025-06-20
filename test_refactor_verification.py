#!/usr/bin/env python3
"""
Refactor Verification Test
=========================

Tests to verify that all critical flake8 errors have been resolved
and the codebase is working correctly after the refactor.
"""

import sys
import os
import importlib
from typing import List, Dict, Any

def test_imports() -> Dict[str, bool]:
    """Test that all core modules can be imported without errors"""
    results = {}
    
    # Test core package imports
    try:
        import core
        results['core'] = True
        print("✅ Core package imports successfully")
    except Exception as e:
        results['core'] = False
        print(f"❌ Core package import failed: {e}")
    
    # Test specific module imports
    modules_to_test = [
        'core.constants',
        'core.fault_bus', 
        'core.error_handler',
        'core.filters',
        'core.type_defs',
        'core.import_resolver',
        'core.best_practices_enforcer',
        'tests.hooks.metrics'
    ]
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            results[module_name] = True
            print(f"✅ {module_name} imports successfully")
        except Exception as e:
            results[module_name] = False
            print(f"❌ {module_name} import failed: {e}")
    
    return results

def test_critical_classes() -> Dict[str, bool]:
    """Test that critical classes can be instantiated"""
    results = {}
    
    try:
        from core.constants import WindowsCliCompatibilityHandler
        handler = WindowsCliCompatibilityHandler()
        results['WindowsCliCompatibilityHandler'] = True
        print("✅ WindowsCliCompatibilityHandler instantiated successfully")
    except Exception as e:
        results['WindowsCliCompatibilityHandler'] = False
        print(f"❌ WindowsCliCompatibilityHandler instantiation failed: {e}")
    
    try:
        from core.fault_bus import FaultBus
        fault_bus = FaultBus()
        results['FaultBus'] = True
        print("✅ FaultBus instantiated successfully")
    except Exception as e:
        results['FaultBus'] = False
        print(f"❌ FaultBus instantiation failed: {e}")
    
    try:
        from tests.hooks.metrics import SchwabotMetrics
        metrics = SchwabotMetrics()
        results['SchwabotMetrics'] = True
        print("✅ SchwabotMetrics instantiated successfully")
    except Exception as e:
        results['SchwabotMetrics'] = False
        print(f"❌ SchwabotMetrics instantiation failed: {e}")
    
    return results

def test_basic_functionality() -> Dict[str, bool]:
    """Test basic functionality of key components"""
    results = {}
    
    # Test Windows CLI compatibility
    try:
        from core.constants import WindowsCliCompatibilityHandler
        handler = WindowsCliCompatibilityHandler()
        is_windows = handler.is_windows_cli()
        safe_msg = handler.safe_print("Test message 🚀")
        results['WindowsCliCompatibility'] = True
        print(f"✅ Windows CLI compatibility test passed (is_windows: {is_windows})")
    except Exception as e:
        results['WindowsCliCompatibility'] = False
        print(f"❌ Windows CLI compatibility test failed: {e}")
    
    # Test fault bus basic operations
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
        results['FaultBusOperations'] = True
        print("✅ FaultBus basic operations test passed")
    except Exception as e:
        results['FaultBusOperations'] = False
        print(f"❌ FaultBus basic operations test failed: {e}")
    
    # Test metrics recording
    try:
        from tests.hooks.metrics import SchwabotMetrics
        metrics = SchwabotMetrics()
        metrics.record_zygot_metric('drift_resonance', 0.75)
        results['MetricsRecording'] = True
        print("✅ Metrics recording test passed")
    except Exception as e:
        results['MetricsRecording'] = False
        print(f"❌ Metrics recording test failed: {e}")
    
    return results

def run_flake8_check() -> Dict[str, Any]:
    """Run flake8 to check for remaining critical errors"""
    import subprocess
    
    try:
        # Run flake8 on core and test directories
        result = subprocess.run([
            'flake8', 'core/', 'tests/', 
            '--count', 
            '--select=E999,F821,F722,E302,E305,E501,F401',
            '--max-line-length=79'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                'success': True,
                'errors': 0,
                'output': result.stdout
            }
        else:
            return {
                'success': False,
                'errors': len(result.stdout.splitlines()),
                'output': result.stdout
            }
    except Exception as e:
        return {
            'success': False,
            'errors': -1,
            'output': f"Flake8 check failed: {e}"
        }

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("SCHWABOT REFACTOR VERIFICATION TEST")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing Imports...")
    import_results = test_imports()
    
    # Test class instantiation
    print("\n2. Testing Class Instantiation...")
    class_results = test_critical_classes()
    
    # Test basic functionality
    print("\n3. Testing Basic Functionality...")
    func_results = test_basic_functionality()
    
    # Run flake8 check
    print("\n4. Running Flake8 Check...")
    flake8_results = run_flake8_check()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_imports = len(import_results)
    successful_imports = sum(import_results.values())
    print(f"Imports: {successful_imports}/{total_imports} successful")
    
    total_classes = len(class_results)
    successful_classes = sum(class_results.values())
    print(f"Classes: {successful_classes}/{total_classes} instantiated successfully")
    
    total_funcs = len(func_results)
    successful_funcs = sum(func_results.values())
    print(f"Functions: {successful_funcs}/{total_funcs} working correctly")
    
    if flake8_results['success']:
        print(f"Flake8: ✅ No critical errors found")
    else:
        print(f"Flake8: ❌ {flake8_results['errors']} critical errors found")
        print("Errors:")
        print(flake8_results['output'])
    
    # Overall success
    all_passed = (
        successful_imports == total_imports and
        successful_classes == total_classes and
        successful_funcs == total_funcs and
        flake8_results['success']
    )
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Refactor successful!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 