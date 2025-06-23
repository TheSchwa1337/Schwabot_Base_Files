#!/usr/bin/env python3
"""
Minimal test to check Python functionality.
"""

print("Python is working!")
print("Testing basic imports...")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import json
    print("✅ JSON imported successfully")
except ImportError as e:
    print(f"❌ JSON import failed: {e}")

try:
    import asyncio
    print("✅ Asyncio imported successfully")
except ImportError as e:
    print(f"❌ Asyncio import failed: {e}")

try:
    import hashlib
    print("✅ Hashlib imported successfully")
except ImportError as e:
    print(f"❌ Hashlib import failed: {e}")

print("Basic test completed!") 