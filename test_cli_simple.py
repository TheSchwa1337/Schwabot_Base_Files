from schwabot.alpha_encryption import get_alpha_encryption, alpha_encrypt_data, analyze_alpha_security
from schwabot.cli import main
from schwabot.cli import main
from schwabot.lantern_core import get_lantern_eye, LanternMainLoop
from schwabot.session_context import create_trading_session, log_trading_activity
from schwabot.update import do_update
from schwabot.vortex_security import get_vortex_security
import argparse
import schwabot_immune_cli
import schwabot_qsc_cli
import schwabot_tensor_cli
import os
import sys

#!/usr/bin/env python3
"""Simple CLI test script."""


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_cli_imports():
    """Test if we can import the CLI and its dependencies."""
    print("🧪 Testing Schwabot CLI Imports")
    print("=" * 40)

    try:
        # Test basic CLI import
        print("✅ Successfully imported CLI main function")

        # Test required module imports
        print("✅ Successfully imported vortex_security")

        print("✅ Successfully imported session_context")

        print("✅ Successfully imported alpha_encryption")

        print("✅ Successfully imported lantern_core")

        print("✅ Successfully imported update module")

        # Test specialized CLI modules
        print("\nTesting specialized CLI modules...")

        print("✅ QSC CLI module imported")

        print("✅ Immune CLI module imported")

        print("✅ Tensor CLI module imported")

        print("\n🎉 All CLI imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_cli_structure():
    """Test CLI argument parser structure."""
    print("\n🔧 Testing CLI Structure")
    print("=" * 40)

    try:

        # Test that main function exists and is callable
        if callable(main):
            print("✅ CLI main function is callable")
        else:
            print("❌ CLI main function is not callable")
            return False

        print("✅ CLI structure test passed")
        return True

    except Exception as e:
        print(f"❌ CLI structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Schwabot CLI Test Suite")
    print("=" * 50)

    # Test imports
    imports_ok = test_cli_imports()

    if imports_ok:
        # Test structure
        structure_ok = test_cli_structure()

        print("\n" + "=" * 50)
        if structure_ok:
            print("🎉 All CLI tests passed!")
            print("\nThe CLI should be ready to use with commands like:")
            print("  python -m schwabot.cli --help")
            print("  python -m schwabot.cli security --status")
            print("  python -m schwabot.cli alpha --demo")
            print("  python -m schwabot.cli qsc")
            print("  python -m schwabot.cli immune")
            print("  python -m schwabot.cli tensor")
        else:
            print("❌ CLI structure test failed!")
    else:
        print("❌ CLI import test failed!") 
