# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
# #!/usr/bin/env python3
"""
Simple Import Test - Check Critical Module Imports and Runtime
"""

import sys
import traceback

safe_print("Starting import and runtime test...")

try:
    from hash_registry_manager import HashRegistryManager
    safe_print("✓ HashRegistryManager import - SUCCESS")
    mgr = HashRegistryManager()
    safe_print("✓ HashRegistryManager instantiation - SUCCESS")
    # Try calling a method that would be used in the integration test
    safe_print("Testing get_hash_entry('hash_00'):")
    entry = mgr.get_hash_entry('hash_00')
    safe_print(f"Result: {entry}")
    safe_print("Testing get_registry_statistics():")
    stats = mgr.get_registry_statistics()
    safe_print(f"Stats: {stats}")
    safe_print("✓ HashRegistryManager runtime test - SUCCESS")
except Exception as e:
    safe_print(f"✗ HashRegistryManager runtime test - FAILED: {e}")
    traceback.print_exc()

def test_import(module_name, class_name=None):
    """Test importing a module and optionally a class."""
    try:
        if class_name:
            exec(f"from {module_name} import {class_name}")
            safe_print(f"✓ {module_name}.{class_name} - SUCCESS")
            return True
        else:
            exec(f"import {module_name}")
            safe_print(f"✓ {module_name} - SUCCESS")
            return True
    except Exception as e:
        safe_print(f"✗ {module_name}{'.' + class_name if class_name else ''} - FAILED: {e}")
        return False

def main():
    """Test all critical imports."""
    safe_print("Testing Critical Module Imports")
    safe_print("=" * 50)

    # Test the modules we just created
    tests = [
        ("profit_bridge_orchestrator", "ProfitBridgeOrchestrator"),
        ("component_registry", "ComponentRegistry"),
        ("unified_signal_metrics", "TradingSignalMetrics"),
        ("unified_signal_metrics", "BTCInvestmentSignals"),
        ("unified_signal_metrics", "collect_unified_signals"),
    ]

    passed = 0
    total = len(tests)

    for module, class_name in tests:
        if test_import(module, class_name):
            passed += 1

    safe_print("=" * 50)
    safe_print(f"Results: {passed}/{total} imports successful")

    if passed == total:
        safe_print("All critical modules imported successfully!")
        return True
    else:
        safe_print("Some imports failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
