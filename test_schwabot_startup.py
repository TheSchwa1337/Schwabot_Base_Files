#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Schwabot Startup Sequence

Simple test to verify the Schwabot startup sequence works correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_schwabot_startup():
    """Test the Schwabot startup sequence."""
    print("🧪 Testing Schwabot AI startup sequence...")
    
    try:
        # Run schwabot.py with --startup flag
        result = subprocess.run(
            [sys.executable, "schwabot.py", "--startup"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Schwabot startup test passed!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Schwabot startup test failed!")
            print("Error:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Test error: {e}")

def test_schwabot_status():
    """Test the Schwabot status command."""
    print("\n🧪 Testing Schwabot AI status...")
    
    try:
        # Run schwabot.py with --status flag
        result = subprocess.run(
            [sys.executable, "schwabot.py", "--status"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Schwabot status test passed!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Schwabot status test failed!")
            print("Error:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_schwabot_startup()
    test_schwabot_status() 