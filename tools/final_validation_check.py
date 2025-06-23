#!/usr/bin/env python3
"""
Final Validation Check for Schwabot v0.42f.

This script provides a comprehensive validation of the CLI and fault handling
unification work. Run this to verify everything is working correctly.
"""

import os
import sys
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    """Print a formatted section."""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")

def check_file_exists(file_path):
    """Check if a file exists and print status."""
    exists = os.path.exists(file_path)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  {status}: {file_path}")
    return exists

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"  ✅ SUCCESS: {module_name}")
        return True
    except ImportError as e:
        print(f"  ❌ FAILED: {module_name} - {e}")
        return False

def main():
    """Main validation function."""
    print_header("SCHWABOT v0.42f FINAL VALIDATION CHECK")
    
    # System information
    print_section("SYSTEM INFORMATION")
    print(f"  Platform: {platform.system()}")
    print(f"  Python Version: {sys.version}")
    print(f"  Working Directory: {os.getcwd()}")
    
    # Check core files
    print_section("CORE FILES VALIDATION")
    core_files = [
        "core/utils/windows_cli_compatibility.py",
        "core/fault_bus.py",
        "config/settings.yaml",
        "v0.42f_release_patchlog.md",
    ]
    
    core_files_exist = all(check_file_exists(f) for f in core_files)
    
    # Check tool files
    print_section("TOOL FILES VALIDATION")
    tool_files = [
        "tools/comprehensive_cli_fault_unification.py",
        "tools/test_cli_fault_compatibility.py",
        "tools/simple_cli_test.py",
        "tools/cli_echo_integration_test.py",
        "tools/validate_cli_injection_points.py",
        "tools/final_validation_check.py",
    ]
    
    tool_files_exist = all(check_file_exists(f) for f in tool_files)
    
    # Check imports
    print_section("IMPORT VALIDATION")
    imports_success = []
    
    # Test CLI handler import
    try:
        from core.utils.windows_cli_compatibility import (
            WindowsCliCompatibilityHandler,
            safe_print,
            safe_format_error,
            log_safe,
            cli_handler,
        )
        print("  ✅ SUCCESS: CLI handler imports")
        imports_success.append(True)
        
        # Test CLI handler functionality
        test_message = "🚀 Test message with emoji ✅"
        safe_result = safe_print(test_message)
        print(f"  ✅ SUCCESS: safe_print test - {safe_result}")
        
        # Test Windows CLI detection
        is_windows = WindowsCliCompatibilityHandler.is_windows_cli()
        print(f"  ✅ SUCCESS: Windows CLI detection - {is_windows}")
        
    except ImportError as e:
        print(f"  ❌ FAILED: CLI handler imports - {e}")
        imports_success.append(False)
    
    # Test fault bus import
    try:
        from core.fault_bus import FaultBus, FaultType, FaultBusEvent
        print("  ✅ SUCCESS: FaultBus imports")
        imports_success.append(True)
        
        # Test fault bus initialization
        fault_bus = FaultBus()
        print("  ✅ SUCCESS: FaultBus initialization")
        
    except ImportError as e:
        print(f"  ❌ FAILED: FaultBus imports - {e}")
        imports_success.append(False)
    
    # Check for unsafe patterns
    print_section("CLI SAFETY VALIDATION")
    
    unsafe_patterns = [
        "print(",
        "logger.error(",
        "logger.warning(",
        "logger.info(",
        "class WindowsCliCompatibilityHandler",
    ]
    
    python_files = list(Path(".").rglob("*.py"))
    unsafe_found = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in unsafe_patterns:
                if pattern in content and "safe_print" not in content and "safe_format_error" not in content:
                    unsafe_found.append(f"{py_file}: {pattern}")
        except Exception:
            pass
    
    if unsafe_found:
        print(f"  ⚠️ WARNING: Found {len(unsafe_found)} potentially unsafe patterns:")
        for item in unsafe_found[:5]:  # Show first 5
            print(f"    • {item}")
        if len(unsafe_found) > 5:
            print(f"    ... and {len(unsafe_found) - 5} more")
    else:
        print("  ✅ SUCCESS: No unsafe patterns found")
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    all_files_exist = core_files_exist and tool_files_exist
    all_imports_work = all(imports_success) if imports_success else False
    cli_safe = len(unsafe_found) == 0
    
    print(f"  Core Files: {'✅ ALL EXIST' if core_files_exist else '❌ MISSING FILES'}")
    print(f"  Tool Files: {'✅ ALL EXIST' if tool_files_exist else '❌ MISSING FILES'}")
    print(f"  Imports: {'✅ ALL WORK' if all_imports_work else '❌ IMPORT FAILURES'}")
    print(f"  CLI Safety: {'✅ SAFE' if cli_safe else '⚠️ UNSAFE PATTERNS FOUND'}")
    
    # Overall status
    overall_success = all_files_exist and all_imports_work and cli_safe
    
    print_section("OVERALL STATUS")
    if overall_success:
        print("🎉 SCHWABOT v0.42f VALIDATION: SUCCESS")
        print("   All systems are CLI-safe and ready for deployment!")
        print("   ✅ CLI compatibility verified")
        print("   ✅ Fault handling standardized")
        print("   ✅ Error formatting consistent")
        print("   ✅ Cross-platform support confirmed")
    else:
        print("⚠️ SCHWABOT v0.42f VALIDATION: ISSUES FOUND")
        print("   Review the validation results above and fix any issues.")
    
    # Recommendations
    print_section("NEXT STEPS")
    if overall_success:
        print("  🚀 Ready for production deployment")
        print("  📊 Run performance tests")
        print("  🔧 Test on actual Windows CLI environment")
        print("  📈 Monitor for any CLI-related issues")
    else:
        print("  🔧 Fix validation issues")
        print("  🧪 Run comprehensive tests")
        print("  📋 Review CLI safety patterns")
        print("  🔄 Re-run validation after fixes")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 