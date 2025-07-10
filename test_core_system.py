#!/usr/bin/env python3
"""
Core System Test
================

Simple test to verify that the Schwabot core system can be imported
and initialized properly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_core_system_import():
    """Test that the core system can be imported."""
    print("Testing core system imports...")
    
    try:
        from core.schwabot_core_system import SchwabotCoreSystem
        print("✅ SchwabotCoreSystem imported successfully")
        return True
    except Exception as e:
        print(f"❌ SchwabotCoreSystem import failed: {e}")
        return False


async def test_core_system_initialization():
    """Test that the core system can be initialized."""
    print("\nTesting core system initialization...")
    
    try:
        from core.schwabot_core_system import SchwabotCoreSystem

        # Create system instance
        system = SchwabotCoreSystem()
        print("✅ SchwabotCoreSystem instance created")
        
        # Test initialization
        success = await system.initialize()
        if success:
            print("✅ SchwabotCoreSystem initialized successfully")
        else:
            print("❌ SchwabotCoreSystem initialization failed")
            return False
        
        # Test system status
        status = system.get_system_status()
        print(f"✅ System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core system initialization failed: {e}")
        return False


async def test_strategy_components():
    """Test strategy components."""
    print("\nTesting strategy components...")
    
    try:
        from core.strategy.strategy_executor import StrategyExecutor
        from core.strategy.strategy_loader import StrategyLoader

        # Test strategy loader
        loader = StrategyLoader()
        success = await loader.initialize()
        if success:
            print("✅ StrategyLoader initialized successfully")
        else:
            print("❌ StrategyLoader initialization failed")
            return False
        
        # Test strategy executor
        executor = StrategyExecutor()
        success = await executor.initialize()
        if success:
            print("✅ StrategyExecutor initialized successfully")
        else:
            print("❌ StrategyExecutor initialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy components test failed: {e}")
        return False


async def test_trading_components():
    """Test trading components."""
    print("\nTesting trading components...")
    
    try:
        from core.btc_usdc_trading_engine import BTCTradingEngine
        from core.risk_manager import RiskManager
        from core.secure_exchange_manager import SecureExchangeManager

        # Test trading engine
        engine = BTCTradingEngine(config={
            "api_key": "demo",
            "api_secret": "demo",
            "testnet": True
        })
        success = await engine.initialize()
        if success:
            print("✅ BTCTradingEngine initialized successfully")
        else:
            print("❌ BTCTradingEngine initialization failed")
            return False
        
        # Test risk manager
        risk_manager = RiskManager()
        success = await risk_manager.initialize()
        if success:
            print("✅ RiskManager initialized successfully")
        else:
            print("❌ RiskManager initialization failed")
            return False
        
        # Test exchange manager
        exchange_manager = SecureExchangeManager()
        success = await exchange_manager.initialize()
        if success:
            print("✅ SecureExchangeManager initialized successfully")
        else:
            print("❌ SecureExchangeManager initialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Trading components test failed: {e}")
        return False


async def test_mathematical_components():
    """Test mathematical components."""
    print("\nTesting mathematical components...")
    
    try:
        from core.enhanced_mathematical_core import EnhancedMathematicalCore
        from core.math_config_manager import MathConfigManager

        # Test math config manager
        math_config = MathConfigManager()
        success = await math_config.initialize()
        if success:
            print("✅ MathConfigManager initialized successfully")
        else:
            print("❌ MathConfigManager initialization failed")
            return False
        
        # Test enhanced mathematical core
        math_core = EnhancedMathematicalCore()
        success = await math_core.initialize()
        if success:
            print("✅ EnhancedMathematicalCore initialized successfully")
        else:
            print("❌ EnhancedMathematicalCore initialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical components test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Schwabot Core System Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_core_system_import),
        ("Initialization Test", test_core_system_initialization),
        ("Strategy Components Test", test_strategy_components),
        ("Trading Components Test", test_trading_components),
        ("Mathematical Components Test", test_mathematical_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core system tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 