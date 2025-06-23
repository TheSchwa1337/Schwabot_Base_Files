#!/usr/bin/env python3
"""
Simple Import Test
=================

Very simple test to check basic imports without circular dependencies.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_imports():
    """Test simple imports."""
    print("Testing simple imports...")
    
    try:
        # Test 1: Basic type_defs import
        print("1. Testing type_defs import...")
        from core.type_defs import BitLevel, MatrixPhase
        print("   PASS: BitLevel and MatrixPhase imported")
        
        # Test 2: MatrixController import
        print("2. Testing MatrixController import...")
        from core.type_defs import MatrixController
        print("   PASS: MatrixController imported")
        
        # Test 3: Create a simple controller
        print("3. Testing controller creation...")
        controller = MatrixController(
            bit_level=BitLevel.FOUR_BIT,
            phase=MatrixPhase.INITIALIZATION,
            hash_signature="test_hash"
        )
        print("   PASS: MatrixController created successfully")
        
        print("\nAll basic imports successful!")
        return True
        
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_imports()
    sys.exit(0 if success else 1) 