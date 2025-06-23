#!/usr/bin/env python3
"""
Basic Import Test - Schwabot UROS v1.0
=====================================

Simple test to check which core modules can be imported successfully.
This helps identify import issues and circular dependencies.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports to identify issues."""
    print("Testing basic imports for Schwabot UROS v1.0")
    print("=" * 50)
    
    # Test core modules
    modules_to_test = [
        ("type_defs", "core.type_defs"),
        ("fault_bus", "core.fault_bus"),
        ("hash_confidence_evaluator", "core.hash_confidence_evaluator"),
        ("unified_confidence_matrix", "core.unified_confidence_matrix"),
        ("mathematical_pipeline_validator_simple", "core.mathematical_pipeline_validator_simple")
    ]
    
    results = {}
    
    for module_name, import_path in modules_to_test:
        try:
            print(f"Testing import: {module_name}...")
            module = __import__(import_path, fromlist=['*'])
            print(f"  PASS: {module_name} imported successfully")
            results[module_name] = "PASS"
        except ImportError as e:
            print(f"  FAIL: {module_name} import failed - {e}")
            results[module_name] = f"FAIL: {e}"
        except Exception as e:
            print(f"  ERROR: {module_name} unexpected error - {e}")
            results[module_name] = f"ERROR: {e}"
    
    print("\n" + "=" * 50)
    print("IMPORT TEST RESULTS:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for module_name, result in results.items():
        if result == "PASS":
            print(f"  PASS: {module_name}")
            passed += 1
        else:
            print(f"  FAIL: {module_name} - {result}")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All imports successful! System is ready for validation.")
        return True
    else:
        print("Some imports failed. Please fix import issues before proceeding.")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 