#!/usr/bin/env python3
"""
Simple test script to debug CLI output issues.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from utils.safe_print import error, info, safe_print, success, warn
    print("✅ Safe print imports successful")

    # Test safe print functions
    safe_print("Test safe_print")
    info("Test info")
    warn("Test warn")
    error("Test error")
    success("Test success")

    print("✅ All safe print functions working")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

print("Test script completed") 