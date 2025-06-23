#!/usr/bin/env python3
"""
Simple import test for UROS v1.0 modules.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except Exception as e:
        print(f"❌ {description}: {module_name} - {e}")
        traceback.print_exc()
        return False

def main():
    """Test all required imports."""
    print("Testing UROS v1.0 module imports...")
    print("=" * 50)
    
    tests = [
        ("core.gpt_command_layer", "GPT Command Layer"),
        ("core.prophet_connector", "Prophet Connector"),
        ("core.memory_stack.ai_command_sequencer", "AI Command Sequencer"),
        ("core.memory_stack.memory_key_allocator", "Memory Key Allocator"),
        ("core.memory_stack.execution_validator", "Execution Validator"),
        ("core.strategy_mapper", "Strategy Mapper"),
        ("core.dlt_waveform_engine", "DLT Waveform Engine"),
        ("core.hash_registry", "Hash Registry"),
        ("core.api_gateway", "API Gateway"),
        ("core.fault_bus", "Fault Bus"),
        ("core.utils.windows_cli_compatibility", "Windows CLI Compatibility"),
    ]
    
    results = []
    for module, description in tests:
        success = test_import(module, description)
        results.append((module, description, success))
    
    print("\n" + "=" * 50)
    print("Import Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All imports successful!")
        return True
    else:
        print("⚠️ Some imports failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 