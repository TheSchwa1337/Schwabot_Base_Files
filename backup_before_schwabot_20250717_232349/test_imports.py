#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple import test to isolate indentation errors.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test each import individually to find the problematic one."""
    
    print("Testing imports one by one...")
    
    # Test 1: Hash Config Manager
    try:
        print("Testing core.hash_config_manager...")
        from core.hash_config_manager import HashConfigManager, get_hash_settings
        print("✅ core.hash_config_manager - OK")
    except Exception as e:
        print(f"❌ core.hash_config_manager - FAILED: {e}")
    
    # Test 2: MathLib
    try:
        print("Testing mathlib...")
        from mathlib import MathLib
        print("✅ mathlib - OK")
    except Exception as e:
        print(f"❌ mathlib - FAILED: {e}")
    
    # Test 3: Quantum Strategy
    try:
        print("Testing mathlib.quantum_strategy...")
        from mathlib.quantum_strategy import QuantumStrategyEngine
        print("✅ mathlib.quantum_strategy - OK")
    except Exception as e:
        print(f"❌ mathlib.quantum_strategy - FAILED: {e}")
    
    # Test 4: Persistent Homology
    try:
        print("Testing mathlib.persistent_homology...")
        from mathlib.persistent_homology import PersistentHomology
        print("✅ mathlib.persistent_homology - OK")
    except Exception as e:
        print(f"❌ mathlib.persistent_homology - FAILED: {e}")
    
    # Test 5: Phantom Band Navigator
    try:
        print("Testing strategies.phantom_band_navigator...")
        from strategies.phantom_band_navigator import PhantomBandNavigator
        print("✅ strategies.phantom_band_navigator - OK")
    except Exception as e:
        print(f"❌ strategies.phantom_band_navigator - FAILED: {e}")

if __name__ == "__main__":
    test_imports() 