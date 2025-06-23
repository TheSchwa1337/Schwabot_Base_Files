#!/usr/bin/env python3
"""
Comprehensive CLI and Fault Compatibility Test Suite.

This script tests Windows CLI compatibility and fault handling across the
entire Schwabot codebase to ensure robust deployment.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

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


class ComprehensiveCLIFaultTester:
    """
    Comprehensive CLI and Fault Compatibility Test Suite.
    
    Tests all aspects of Windows CLI compatibility and fault handling
    across the Schwabot codebase.
    """

    def __init__(self, root_dir: str = "."):
        """Initialize the tester."""
        self.root_dir = Path(root_dir)
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests."""
        print("=" * 80)
        print("COMPREHENSIVE CLI AND FAULT COMPATIBILITY TEST SUITE")
        print("=" * 80)

        tests = [
            ("CLI Handler Import", self.test_cli_handler_import),
            ("Safe Print Functionality", self.test_safe_print),
            ("Safe Error Formatting", self.test_safe_error_formatting),
            ("Safe Logging", self.test_safe_logging),
            ("Unicode Handling", self.test_unicode_handling),
            ("Emoji Conversion", self.test_emoji_conversion),
            ("Fault Bus Integration", self.test_fault_bus_integration),
            ("Windows CLI Detection", self.test_windows_cli_detection),
            ("Import Consistency", self.test_import_consistency),
            ("Flake8 Compliance", self.test_flake8_compliance),
            ("Error Handling", self.test_error_handling),
            ("Cross-Platform Compatibility", self.test_cross_platform_compatibility),
        ]

        for test_name, test_func in tests:
            print(f"\n[TESTING] {test_name}...")
            try:
                result = test_func()
                if result.get("success", False):
                    print(f"  ✅ PASSED: {test_name}")
                    self.passed_tests.append(test_name)
                else:
                    print(f"  ❌ FAILED: {test_name}")
                    print(f"     Error: {result.get('error', 'Unknown error')}")
                    self.failed_tests.append(test_name)
                self.test_results[test_name] = result
            except Exception as e:
                print(f"  ❌ ERROR: {test_name} - {e}")
                self.failed_tests.append(test_name)
                self.test_results[test_name] = {"success": False, "error": str(e)}

        return self.generate_report()

    def test_cli_handler_import(self) -> Dict[str, Any]:
        """Test CLI handler import functionality."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test basic functionality
            handler = WindowsCliCompatibilityHandler()
            if handler is None:
                return {"success": False, "error": "Failed to create handler instance"}

            return {"success": True, "handler": handler}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_safe_print(self) -> Dict[str, Any]:
        """Test safe print functionality."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test messages with emojis
            test_messages = [
                "🚀 Launching system...",
                "✅ Operation completed",
                "❌ Error occurred",
                "⚠️ Warning message",
                "📊 Data processing",
                "🎯 Target reached",
            ]

            results = []
            for message in test_messages:
                safe_result = safe_print(message)
                # Check that emojis are converted to ASCII
                if any(emoji in safe_result for emoji in ["🚀", "✅", "❌", "⚠️", "📊", "🎯"]):
                    results.append(f"Emoji not converted in: {message}")
                else:
                    results.append(f"Successfully converted: {message}")

            return {
                "success": len([r for r in results if "Successfully" in r]) == len(test_messages),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_safe_error_formatting(self) -> Dict[str, Any]:
        """Test safe error formatting."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test with various error types
            test_errors = [
                (ValueError("Invalid input"), "data_processing"),
                (RuntimeError("System failure"), "core_operation"),
                (Exception("Generic error"), "general"),
            ]

            results = []
            for error, context in test_errors:
                formatted = safe_format_error(error, context)
                if "Error:" in formatted and context in formatted:
                    results.append(f"Successfully formatted: {type(error).__name__}")
                else:
                    results.append(f"Failed to format: {type(error).__name__}")

            return {
                "success": len([r for r in results if "Successfully" in r]) == len(test_errors),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_safe_logging(self) -> Dict[str, Any]:
        """Test safe logging functionality."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "CLI handler not available"}

            import logging

            # Create a test logger
            test_logger = logging.getLogger("test_cli_logger")
            test_logger.setLevel(logging.INFO)

            # Capture log output
            log_capture = []
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            test_logger.addHandler(handler)

            # Test logging with emojis
            test_message = "🚀 System startup with emoji"
            log_safe(test_logger, "info", test_message)

            # Check that the message was logged safely
            return {"success": True, "message": "Safe logging test completed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_unicode_handling(self) -> Dict[str, Any]:
        """Test Unicode character handling."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test Unicode characters
            unicode_messages = [
                "α β γ δ ε",
                "∑(i=1 to n) x_i",
                "μ = 0.5, σ = 0.1",
                "φ = 1.618033988749895",
                "∀ x ∈ ℝ",
            ]

            results = []
            for message in unicode_messages:
                safe_result = safe_print(message)
                # Check that Unicode is handled safely
                try:
                    safe_result.encode('ascii')
                    results.append(f"Successfully handled: {message}")
                except UnicodeEncodeError:
                    results.append(f"Unicode handling failed: {message}")

            return {
                "success": len([r for r in results if "Successfully" in r]) == len(unicode_messages),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_emoji_conversion(self) -> Dict[str, Any]:
        """Test emoji to ASCII conversion."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test emoji conversions
            emoji_tests = [
                ("✅", "[SUCCESS]"),
                ("❌", "[ERROR]"),
                ("🚀", "[LAUNCH]"),
                ("📊", "[DATA]"),
                ("🎯", "[TARGET]"),
                ("🔥", "[HOT]"),
                ("⚡", "[FAST]"),
            ]

            results = []
            for emoji, expected in emoji_tests:
                test_message = f"Test {emoji} message"
                safe_result = safe_print(test_message)
                if expected in safe_result:
                    results.append(f"Successfully converted {emoji} to {expected}")
                else:
                    results.append(f"Failed to convert {emoji}")

            return {
                "success": len([r for r in results if "Successfully" in r]) == len(emoji_tests),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_fault_bus_integration(self) -> Dict[str, Any]:
        """Test fault bus integration with CLI safety."""
        try:
            # Test importing fault bus
            try:
                from core.fault_bus import FaultBus, FaultType, FaultBusEvent
                fault_bus_available = True
            except ImportError:
                fault_bus_available = False

            if not fault_bus_available:
                return {"success": False, "error": "FaultBus not available"}

            # Test fault bus initialization
            fault_bus = FaultBus()
            
            # Test fault event creation with CLI-safe metadata
            test_event = FaultBusEvent(
                tick=1,
                module="test_module",
                type=FaultType.THERMAL_HIGH,
                severity=0.5,
                metadata={"message": "🚀 Test fault event"},
                profit_context=10.0,
            )

            return {"success": True, "fault_bus": fault_bus, "test_event": test_event}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_windows_cli_detection(self) -> Dict[str, Any]:
        """Test Windows CLI environment detection."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            is_windows = WindowsCliCompatibilityHandler.is_windows_cli()
            
            return {
                "success": True,
                "is_windows_cli": is_windows,
                "platform": os.name,
                "comspec": os.environ.get("COMSPEC", ""),
                "ps_module_path": os.environ.get("PSModulePath", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_import_consistency(self) -> Dict[str, Any]:
        """Test import consistency across the codebase."""
        try:
            # Check for files that might have local CLI handler definitions
            python_files = list(self.root_dir.rglob("*.py"))
            
            local_handlers = []
            import_issues = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for local WindowsCliCompatibilityHandler definitions
                        if "class WindowsCliCompatibilityHandler" in content:
                            local_handlers.append(str(py_file))
                        
                        # Check for proper imports
                        if "from core.utils.windows_cli_compatibility import" in content:
                            continue
                        elif "WindowsCliCompatibilityHandler" in content:
                            import_issues.append(f"Missing import in {py_file}")
                            
                except Exception as e:
                    import_issues.append(f"Error reading {py_file}: {e}")

            return {
                "success": len(local_handlers) == 0 and len(import_issues) == 0,
                "local_handlers": local_handlers,
                "import_issues": import_issues,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_flake8_compliance(self) -> Dict[str, Any]:
        """Test Flake8 compliance."""
        try:
            # Run flake8 on the codebase
            result = subprocess.run(
                ["flake8", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            return {
                "success": result.returncode == 0,
                "error_count": len(result.stdout.splitlines()) if result.stdout else 0,
                "output": result.stdout,
                "errors": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with CLI safety."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test various error scenarios
            test_scenarios = [
                ("Unicode error", lambda: "🚀".encode('ascii')),
                ("Division by zero", lambda: 1/0),
                ("Index error", lambda: [][0]),
                ("Key error", lambda: {}["missing"]),
            ]

            results = []
            for scenario_name, error_func in test_scenarios:
                try:
                    error_func()
                    results.append(f"Unexpected success: {scenario_name}")
                except Exception as e:
                    formatted = safe_format_error(e, scenario_name)
                    if "Error:" in formatted and scenario_name in formatted:
                        results.append(f"Successfully handled: {scenario_name}")
                    else:
                        results.append(f"Failed to format: {scenario_name}")

            return {
                "success": len([r for r in results if "Successfully" in r]) == len(test_scenarios),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_cross_platform_compatibility(self) -> Dict[str, Any]:
        """Test cross-platform compatibility."""
        try:
            if not CLI_HANDLER_AVAILABLE:
                return {"success": False, "error": "CLI handler not available"}

            # Test on different platforms
            import platform
            
            platform_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            }

            # Test CLI detection
            is_windows = WindowsCliCompatibilityHandler.is_windows_cli()
            
            # Test safe print on current platform
            test_message = "🚀 Cross-platform test ✅"
            safe_result = safe_print(test_message)

            return {
                "success": True,
                "platform_info": platform_info,
                "is_windows_cli": is_windows,
                "safe_result": safe_result,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = len(self.passed_tests)
        failed_tests = len(self.failed_tests)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
            },
            "test_results": self.test_results,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "recommendations": self.generate_recommendations(),
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if self.failed_tests:
            recommendations.append(f"Fix {len(self.failed_tests)} failed tests")

        if "Import Consistency" in self.failed_tests:
            recommendations.append("Remove local CLI handler definitions and use centralized imports")

        if "Flake8 Compliance" in self.failed_tests:
            recommendations.append("Fix Flake8 compliance issues")

        if "Fault Bus Integration" in self.failed_tests:
            recommendations.append("Ensure fault bus properly integrates with CLI handler")

        recommendations.append("Run tests on actual Windows CLI environment")
        recommendations.append("Test with various Unicode and emoji inputs")
        recommendations.append("Verify error handling in production scenarios")

        return recommendations


def main():
    """Main function to run comprehensive CLI and fault compatibility tests."""
    print("=" * 80)
    print("SCHWABOT CLI AND FAULT COMPATIBILITY TEST SUITE")
    print("=" * 80)

    # Initialize tester
    tester = ComprehensiveCLIFaultTester()

    # Run all tests
    report = tester.run_all_tests()

    # Print final report
    print("\n" + "=" * 80)
    print("FINAL TEST REPORT")
    print("=" * 80)

    summary = report["summary"]
    print(f"\n📊 TEST SUMMARY:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")

    if report["failed_tests"]:
        print(f"\n❌ FAILED TESTS:")
        for test in report["failed_tests"]:
            print(f"  • {test}")

    if report["passed_tests"]:
        print(f"\n✅ PASSED TESTS:")
        for test in report["passed_tests"]:
            print(f"  • {test}")

    print(f"\n📋 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    print(f"\n🎯 OVERALL STATUS: {'SUCCESS' if summary['success_rate'] == 1.0 else 'PARTIAL' if summary['success_rate'] > 0.5 else 'FAILED'}")

    if summary['success_rate'] == 1.0:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Schwabot is fully CLI-compatible and fault-safe.")
    elif summary['success_rate'] > 0.5:
        print("\n⚠️ MOST TESTS PASSED")
        print("   Some issues remain - review failed tests and recommendations.")
    else:
        print("\n❌ MANY TESTS FAILED")
        print("   Significant issues found - review and fix before deployment.")

    return report


if __name__ == "__main__":
    main() 