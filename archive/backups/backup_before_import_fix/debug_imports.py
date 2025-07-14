import sys
import traceback

from dual_unicore_handler import DualUnicoreHandler
from utils.safe_print import debug, error, info, safe_print, success, warn

# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-
"""Debug script to isolate import issues.""""""
""""""
""""""
""""""
"""


def test_import(module_name): """
    """Test importing a specific module."""

"""


""""""
""""""
""""""
"""
   try:"""
safe_print(f"Testing import of {module_name}...")
        __import__(module_name)
        safe_print(f"\\u2705 {module_name} imported successfully")
        return True
    except Exception as e:
        safe_print(f"\\u274c {module_name} import failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Test imports step by step."""


"""
""""""
""""""
""""""
""""""
   safe_print("Debugging import issues...")
    safe_print("=" * 50)

# Test core imports
test_import("core.typing_schemas")
    test_import("core.fault_bus")
    test_import("core.type_binding_system")

# Test UI bridge imports
test_import("core.ui_state_bridge")
    test_import("core.visual_integration_bridge")
    test_import("core.ui_integration_bridge")
    test_import("core.ui_bridge_integration_manager")

# Test trading system imports
test_import("core.ghost_profit_tracker")
    test_import("core.unified_mathematical_trading_controller")
    test_import("core.state_tracker")

# Test core module import
test_import("core")


if __name__ == "__main__":
    main()

""""""
""""""
""""""
""""""
""""""
"""
"""
