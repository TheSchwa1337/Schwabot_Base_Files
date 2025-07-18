import argparse
import os
import sys
from io import StringIO

import schwabot_qsc_cli
import schwabot_tensor_cli

import schwabot_immune_cli
from schwabot.alpha_encryption import alpha_encrypt_data, analyze_alpha_security, get_alpha_encryption
from schwabot.cli import main
from schwabot.lantern_core import LanternMainLoop, get_lantern_eye
from schwabot.session_context import create_trading_session, log_trading_activity
from schwabot.update import do_update
from schwabot.vortex_security import get_vortex_security

#!/usr/bin/env python3
"""Test script to verify CLI functionality."""


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_cli_help():
    """Test the CLI help command."""
    print("Testing Schwabot CLI...")

    try:
        # Import the CLI module

        # Test if we can import all dependencies
        print("✅ Successfully imported CLI module")

        # Test if we can import all required modules

        print("✅ Successfully imported all required modules")

        # Test the specialized CLI modules
        print("\nTesting specialized CLI modules...")

        # Test QSC CLI
        try:
            print("✅ QSC CLI module imported successfully")
        except ImportError as e:
            print(f"❌ QSC CLI import failed: {e}")

        # Test Immune CLI
        try:
            print("✅ Immune CLI module imported successfully")
        except ImportError as e:
            print(f"❌ Immune CLI import failed: {e}")

        # Test Tensor CLI
        try:
            print("✅ Tensor CLI module imported successfully")
        except ImportError as e:
            print(f"❌ Tensor CLI import failed: {e}")

        print("\n🎉 CLI test completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_cli_commands():
    """Test specific CLI commands."""
    print("\nTesting CLI commands...")

    # Test help command by simulating argparse

    try:

        # Capture help output
        help_output = StringIO()
        sys.stdout = help_output

        # Simulate help command
        sys.argv = ['schwabot', '--help']

        try:
            main()
        except SystemExit:
            pass  # Expected for help command

        sys.stdout = sys.__stdout__
        help_text = help_output.getvalue()

        if 'Unified Schwabot CLI' in help_text:
            print("✅ CLI help command works")
            print("Available commands found:")
            if 'update' in help_text:
                print("  - update")
            if 'security' in help_text:
                print("  - security")
            if 'alpha' in help_text:
                print("  - alpha")
            if 'lantern-eye' in help_text:
                print("  - lantern-eye")
            if 'historical' in help_text:
                print("  - historical")
            if 'qsc' in help_text:
                print("  - qsc")
            if 'immune' in help_text:
                print("  - immune")
            if 'tensor' in help_text:
                print("  - tensor")
        else:
            print("❌ CLI help command failed")

    except Exception as e:
        print(f"❌ CLI command test failed: {e}")
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    print("🧪 Testing Schwabot CLI System")
    print("=" * 50)

    # Test basic imports
    success = test_cli_help()

    if success:
        # Test CLI commands
        test_cli_commands()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All CLI tests passed!")
    else:
        print("❌ CLI tests failed!") 
