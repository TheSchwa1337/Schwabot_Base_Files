#!/usr/bin/env python3
"""
Core Functionality Test for Schwabot
===================================
Simple test to validate that the core components work.
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from core.btc_usdc_trading_engine import BTCTradingEngine
        print("✅ BTCTradingEngine imported successfully")
    except Exception as e:
        print(f"❌ BTCTradingEngine import failed: {e}")
        return False
    
    try:
        from core.risk_manager import RiskManager
        print("✅ RiskManager imported successfully")
    except Exception as e:
        print(f"❌ RiskManager import failed: {e}")
        return False
    
    try:
        from core.secure_exchange_manager import SecureExchangeManager
        print("✅ SecureExchangeManager imported successfully")
    except Exception as e:
        print(f"❌ SecureExchangeManager import failed: {e}")
        return False
    
    try:
        from core.unified_pipeline_manager import UnifiedPipelineManager

        # Test instantiation
        pipeline_manager = UnifiedPipelineManager()
        print("✅ UnifiedPipelineManager imported and instantiated successfully")
    except Exception as e:
        print(f"❌ UnifiedPipelineManager import failed: {e}")
        return False
    
    try:
        from core.math_config_manager import MathConfigManager
        print("✅ MathConfigManager imported successfully")
    except Exception as e:
        print(f"❌ MathConfigManager import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        config_path = Path("config/schwabot_config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ Configuration loaded successfully")
            return True
        else:
            print("⚠️  Configuration file not found, but continuing...")
            return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_cli_functionality():
    """Test that CLI can be imported and basic functionality works."""
    print("\nTesting CLI functionality...")
    
    try:
        import main
        print("✅ Main CLI module imported successfully")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_syntax_check():
    """Test that core files have valid Python syntax."""
    print("\nTesting syntax...")
    
    core_files = [
        "main.py",
        "core/__init__.py",
        "core/btc_usdc_trading_engine.py",
        "core/risk_manager.py",
        "core/secure_exchange_manager.py",
        "core/unified_pipeline_manager.py",
        "core/math_config_manager.py"
    ]
    
    all_good = True
    for file_path in core_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path} syntax OK")
        except Exception as e:
            print(f"❌ {file_path} syntax error: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all core functionality tests."""
    print("🚀 Schwabot Core Functionality Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config_loading),
        ("CLI Test", test_cli_functionality),
        ("Syntax Test", test_syntax_check),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 