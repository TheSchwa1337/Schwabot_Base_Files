#!/usr/bin/env python3
"""
Simple CLI test script to verify basic functionality.
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_cli_imports():
    """Test if we can import the CLI components."""
    try:
        print("Testing CLI imports...")
        
        # Test basic imports
        import argparse
        import asyncio
        import logging
        print("✅ Basic imports successful")
        
        # Test core components
        try:
            from core.risk_manager import RiskManager
            print("✅ RiskManager import successful")
        except ImportError as e:
            print(f"⚠️ RiskManager import failed: {e}")
        
        try:
            from core.pure_profit_calculator import PureProfitCalculator
            print("✅ PureProfitCalculator import successful")
        except ImportError as e:
            print(f"⚠️ PureProfitCalculator import failed: {e}")
        
        # Test main CLI
        try:
            from main import SchwabotCLI
            print("✅ SchwabotCLI import successful")
        except ImportError as e:
            print(f"❌ SchwabotCLI import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_cli_initialization():
    """Test CLI initialization."""
    try:
        print("\nTesting CLI initialization...")
        from main import SchwabotCLI
        
        cli = SchwabotCLI()
        print("✅ CLI initialization successful")
        
        # Test basic methods
        help_text = cli.get_help_text()
        if help_text and len(help_text) > 100:
            print("✅ Help text generation successful")
        else:
            print("⚠️ Help text generation may have issues")
        
        platform_info = cli.get_platform_info()
        if platform_info and 'platform' in platform_info:
            print("✅ Platform info generation successful")
        else:
            print("⚠️ Platform info generation may have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI initialization failed: {e}")
        return False

def test_system_status():
    """Test system status functionality."""
    try:
        print("\nTesting system status...")
        from main import SchwabotCLI
        
        cli = SchwabotCLI()
        status = cli.get_system_status()
        
        if status and isinstance(status, dict):
            print("✅ System status generation successful")
            print(f"Status keys: {list(status.keys())}")
            return True
        else:
            print("⚠️ System status may have issues")
            return False
            
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Schwabot CLI Simple Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_cli_imports),
        ("CLI Initialization", test_cli_initialization),
        ("System Status", test_system_status),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLI is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. CLI may have issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
