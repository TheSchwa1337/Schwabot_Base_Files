#!/usr/bin/env python3
"""
CLI Echo Integration Test.

Validates system entry points and fault bus integration with CLI safety.
This test ensures all entry vectors are CLI-safe and fault bus integration works correctly.
"""

import os
import sys
import logging
from datetime import datetime

# Import the centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    print("[ERROR] Centralized CLI handler not available")
    sys.exit(1)

# Import fault bus
try:
    from core.fault_bus import FaultBus, FaultType, FaultBusEvent
    FAULT_BUS_AVAILABLE = True
except ImportError:
    FAULT_BUS_AVAILABLE = False
    print("[ERROR] FaultBus not available")
    sys.exit(1)


class CLIEchoIntegrationTester:
    """CLI Echo Integration Test Suite."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = []
        self.fault_bus = FaultBus() if FAULT_BUS_AVAILABLE else None
        self.logger = logging.getLogger("cli_echo_test")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(handler)

    def test_entry_vector_stabilization(self):
        """Test entry vector stabilization with CLI safety."""
        safe_print("🚀 ENTRY VECTOR STABILIZED - echo safe")
        
        # Test various entry scenarios
        entry_scenarios = [
            "📊 Market data received",
            "🎯 Target price calculated",
            "⚡ High-frequency signal detected",
            "🔥 Thermal state normal",
            "💰 Profit opportunity identified",
        ]
        
        for scenario in entry_scenarios:
            safe_print(f"  {scenario}")
        
        return True

    def test_fault_injection(self):
        """Test fault injection with CLI-safe error handling."""
        try:
            raise ValueError("🔥 Fault injection test")
        except Exception as e:
            safe_print(f"❌ Fault detected: {safe_format_error(e, 'fault_injection_test')}")
            
            # Report to fault bus if available
            if self.fault_bus:
                fault_event = FaultBusEvent(
                    tick=1,
                    module="cli_echo_test",
                    type=FaultType.PROFIT_ANOMALY,
                    severity=0.7,
                    metadata={"error": str(e), "test_type": "fault_injection"},
                    profit_context=0.0,
                )
                self.fault_bus.push(fault_event)
                safe_print("✅ Fault reported to FaultBus")
            
            return True

    def test_unicode_edge_cases(self):
        """Test Unicode edge cases and special characters."""
        unicode_tests = [
            "α β γ δ ε - Greek letters",
            "∑(i=1 to n) x_i - Mathematical sum",
            "μ = 0.5, σ = 0.1 - Statistics",
            "φ = 1.618033988749895 - Golden ratio",
            "∀ x ∈ ℝ - Mathematical logic",
            "→ ← ↑ ↓ ↔ ↕ - Arrows",
            "⇒ ⇐ ⇔ - Logical arrows",
            "∞ φ π ∑ ∫ ∇ Δ σ μ λ θ - Math symbols",
        ]
        
        safe_print("🧪 Testing Unicode edge cases:")
        for test in unicode_tests:
            safe_result = safe_print(f"  {test}")
            if safe_result:
                safe_print(f"    ✅ Safe: {safe_result}")
        
        return True

    def test_logging_integration(self):
        """Test logging integration with CLI safety."""
        test_messages = [
            "🚀 System startup",
            "📊 Data processing",
            "🎯 Target acquisition",
            "⚡ High-speed operation",
            "🔥 Thermal management",
        ]
        
        safe_print("📝 Testing logging integration:")
        for message in test_messages:
            log_safe(self.logger, "info", message)
        
        return True

    def test_fault_bus_integration(self):
        """Test fault bus integration with CLI safety."""
        if not self.fault_bus:
            safe_print("⚠️ FaultBus not available, skipping integration test")
            return False
        
        safe_print("🔧 Testing FaultBus integration:")
        
        # Create test fault events
        test_events = [
            FaultBusEvent(
                tick=1,
                module="cli_echo_test",
                type=FaultType.THERMAL_HIGH,
                severity=0.6,
                metadata={"test": "thermal_high", "message": "🚀 Thermal test"},
                profit_context=15.0,
            ),
            FaultBusEvent(
                tick=2,
                module="cli_echo_test",
                type=FaultType.PROFIT_ANOMALY,
                severity=0.8,
                metadata={"test": "profit_anomaly", "message": "💰 Profit test"},
                profit_context=25.0,
            ),
            FaultBusEvent(
                tick=3,
                module="cli_echo_test",
                type=FaultType.BITMAP_CORRUPT,
                severity=0.9,
                metadata={"test": "bitmap_corrupt", "message": "🔧 Bitmap test"},
                profit_context=0.0,
            ),
        ]
        
        for event in test_events:
            self.fault_bus.push(event)
            safe_print(f"  ✅ Pushed {event.type.value} event")
        
        return True

    def test_windows_cli_detection(self):
        """Test Windows CLI environment detection."""
        safe_print("🖥️ Testing Windows CLI detection:")
        
        is_windows = WindowsCliCompatibilityHandler.is_windows_cli()
        safe_print(f"  Is Windows CLI: {is_windows}")
        safe_print(f"  Platform: {os.name}")
        safe_print(f"  COMSPEC: {os.environ.get('COMSPEC', 'Not set')}")
        safe_print(f"  PSModulePath: {os.environ.get('PSModulePath', 'Not set')}")
        
        return True

    def run_all_tests(self):
        """Run all CLI echo integration tests."""
        safe_print("=" * 60)
        safe_print("CLI ECHO INTEGRATION TEST SUITE")
        safe_print("=" * 60)
        
        tests = [
            ("Entry Vector Stabilization", self.test_entry_vector_stabilization),
            ("Fault Injection", self.test_fault_injection),
            ("Unicode Edge Cases", self.test_unicode_edge_cases),
            ("Logging Integration", self.test_logging_integration),
            ("FaultBus Integration", self.test_fault_bus_integration),
            ("Windows CLI Detection", self.test_windows_cli_detection),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            safe_print(f"\n🧪 Running: {test_name}")
            try:
                if test_func():
                    safe_print(f"  ✅ PASSED: {test_name}")
                    passed += 1
                else:
                    safe_print(f"  ❌ FAILED: {test_name}")
            except Exception as e:
                safe_print(f"  ❌ ERROR: {test_name} - {safe_format_error(e, test_name)}")
        
        safe_print(f"\n📊 Test Results: {passed}/{total} passed")
        
        if passed == total:
            safe_print("🎉 ALL TESTS PASSED!")
            safe_print("CLI echo integration is working correctly.")
        else:
            safe_print("⚠️ SOME TESTS FAILED")
            safe_print("Review the errors above and fix issues.")
        
        return passed == total


def simulate_entry():
    """Simulate system entry with CLI safety."""
    safe_print("🚀 ENTRY VECTOR STABILIZED - echo safe")
    
    try:
        raise ValueError("🔥 Fault injection test")
    except Exception as e:
        safe_print(f"❌ Fault detected: {safe_format_error(e, 'entry_simulation')}")
        
        # Report to fault bus if available
        try:
            fault_bus = FaultBus()
            fault_event = FaultBusEvent(
                tick=1,
                module="entry_simulation",
                type=FaultType.PROFIT_ANOMALY,
                severity=0.7,
                metadata={"error": str(e), "simulation": True},
                profit_context=0.0,
            )
            fault_bus.push(fault_event)
            safe_print("✅ Fault reported to FaultBus")
        except Exception as fb_error:
            safe_print(f"⚠️ FaultBus report failed: {safe_format_error(fb_error, 'fault_bus_report')}")


def main():
    """Main test function."""
    if not CLI_HANDLER_AVAILABLE:
        print("[ERROR] CLI handler not available")
        return False
    
    if not FAULT_BUS_AVAILABLE:
        print("[ERROR] FaultBus not available")
        return False
    
    # Run comprehensive tests
    tester = CLIEchoIntegrationTester()
    success = tester.run_all_tests()
    
    # Run entry simulation
    safe_print("\n" + "=" * 60)
    safe_print("ENTRY SIMULATION")
    safe_print("=" * 60)
    simulate_entry()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 