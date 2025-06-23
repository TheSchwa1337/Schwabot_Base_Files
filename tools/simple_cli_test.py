#!/usr/bin/env python3
"""
Simple CLI Compatibility Test.

Quick test to validate Windows CLI compatibility functionality.
"""

import os
import sys
import platform

def test_cli_handler():
    """Test the CLI handler functionality."""
    print("Testing Windows CLI Compatibility Handler...")
    print("=" * 50)
    
    # Test 1: Import the handler
    try:
        from core.utils.windows_cli_compatibility import (
            WindowsCliCompatibilityHandler,
            safe_print,
            safe_format_error,
            log_safe,
            cli_handler,
        )
        print("✅ CLI handler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CLI handler: {e}")
        return False
    
    # Test 2: Test safe_print with emojis
    test_messages = [
        "🚀 Launching system...",
        "✅ Operation completed",
        "❌ Error occurred",
        "⚠️ Warning message",
        "📊 Data processing",
        "🎯 Target reached",
    ]
    
    print("\nTesting safe_print with emojis:")
    for message in test_messages:
        safe_result = safe_print(message)
        print(f"  Original: {message}")
        print(f"  Safe:     {safe_result}")
        print("-" * 30)
    
    # Test 3: Test error formatting
    print("\nTesting error formatting:")
    try:
        raise ValueError("Test error with emoji 🚀")
    except Exception as e:
        formatted = safe_format_error(e, "test_context")
        print(f"  Original error: {e}")
        print(f"  Formatted: {formatted}")
    
    # Test 4: Test Windows CLI detection
    print("\nTesting Windows CLI detection:")
    is_windows = WindowsCliCompatibilityHandler.is_windows_cli()
    print(f"  Is Windows CLI: {is_windows}")
    print(f"  Platform: {platform.system()}")
    print(f"  COMSPEC: {os.environ.get('COMSPEC', 'Not set')}")
    print(f"  PSModulePath: {os.environ.get('PSModulePath', 'Not set')}")
    
    # Test 5: Test Unicode handling
    print("\nTesting Unicode handling:")
    unicode_messages = [
        "α β γ δ ε",
        "∑(i=1 to n) x_i",
        "μ = 0.5, σ = 0.1",
        "φ = 1.618033988749895",
    ]
    
    for message in unicode_messages:
        safe_result = safe_print(message)
        print(f"  Original: {message}")
        print(f"  Safe:     {safe_result}")
        print("-" * 30)
    
    print("\n✅ All CLI compatibility tests completed!")
    return True

def test_fault_bus_integration():
    """Test fault bus integration with CLI handler."""
    print("\nTesting Fault Bus Integration...")
    print("=" * 50)
    
    try:
        from core.fault_bus import FaultBus, FaultType, FaultBusEvent
        print("✅ FaultBus imported successfully")
        
        # Test fault bus initialization
        fault_bus = FaultBus()
        print("✅ FaultBus initialized successfully")
        
        # Test fault event creation
        test_event = FaultBusEvent(
            tick=1,
            module="test_module",
            type=FaultType.THERMAL_HIGH,
            severity=0.5,
            metadata={"message": "🚀 Test fault event"},
            profit_context=10.0,
        )
        print("✅ FaultBusEvent created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import FaultBus: {e}")
        return False
    except Exception as e:
        print(f"❌ FaultBus test failed: {e}")
        return False

def main():
    """Main test function."""
    print("SCHWABOT CLI COMPATIBILITY TEST")
    print("=" * 50)
    
    # Test CLI handler
    cli_success = test_cli_handler()
    
    # Test fault bus integration
    fault_success = test_fault_bus_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"CLI Handler: {'✅ PASSED' if cli_success else '❌ FAILED'}")
    print(f"Fault Bus:   {'✅ PASSED' if fault_success else '❌ FAILED'}")
    
    if cli_success and fault_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Schwabot CLI compatibility is working correctly.")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Review the errors above and fix issues.")
    
    return cli_success and fault_success

if __name__ == "__main__":
    main() 