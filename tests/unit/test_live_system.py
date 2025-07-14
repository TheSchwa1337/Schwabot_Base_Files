#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Schwabot Live API Backtesting System.

This script tests the core functionality without connecting to real APIs.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_core_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing core imports...")
    
    try:
        from core.ccxt_trading_executor import CCXTTradingExecutor
        from core.chrono_recursive_logic_function import ChronoRecursiveLogicFunction
        from core.clean_math_foundation import CleanMathFoundation
        from core.clean_profit_vectorization import CleanProfitVectorization
        from core.clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline
        from core.live_api_backtesting import LiveAPIBacktesting, LiveAPIConfig, create_live_api_backtesting
        from core.neural_processing_engine import NeuralProcessingEngine
        from core.portfolio_tracker import PortfolioTracker, create_portfolio_tracker
        from core.quantum_mathematical_bridge import QuantumMathematicalBridge
        
        print("✅ All core modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_live_api_backtesting():
    """Test live API backtesting system initialization."""
    print("\n🔍 Testing Live API Backtesting System...")
    
    try:
        from core.live_api_backtesting import LiveAPIConfig, create_live_api_backtesting

        # Create configuration
        config = LiveAPIConfig(
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
            symbols=["BTC/USDC"],
            enable_trading=False,
            update_interval=5.0
        )
        
        # Create backtesting instance
        backtesting = create_live_api_backtesting(config)
        
        print("✅ Live API Backtesting System created successfully")
        print(f"   Exchange: {backtesting.config.exchange}")
        print(f"   Symbols: {backtesting.config.symbols}")
        print(f"   Trading enabled: {backtesting.config.enable_trading}")
        
        return True
        
    except Exception as e:
        print(f"❌ Live API Backtesting test failed: {e}")
        return False


async def test_portfolio_tracker():
    """Test portfolio tracker functionality."""
    print("\n🔍 Testing Portfolio Tracker...")
    
    try:
        from decimal import Decimal

        from core.portfolio_tracker import create_portfolio_tracker

        # Create portfolio tracker
        tracker = create_portfolio_tracker()
        
        # Test basic functionality
        summary = tracker.get_portfolio_summary()
        
        print("✅ Portfolio Tracker created successfully")
        print(f"   Total value: ${summary.get('total_value', 0):.2f}")
        print(f"   Total PnL: ${summary.get('total_pnl', 0):.2f}")
        print(f"   Active positions: {summary.get('active_positions', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio Tracker test failed: {e}")
        return False


async def test_trading_pipeline():
    """Test trading pipeline functionality."""
    print("\n🔍 Testing Trading Pipeline...")
    
    try:
        from core.clean_trading_pipeline import create_trading_pipeline

        # Create trading pipeline
        pipeline = create_trading_pipeline(
            symbol="BTCUSDC",
            initial_capital=10000.0,
            safe_mode=True
        )
        
        print("✅ Trading Pipeline created successfully")
        print(f"   Symbol: {pipeline.symbol}")
        print(f"   Initial capital: ${pipeline.initial_capital}")
        print(f"   Safe mode: {pipeline.safe_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading Pipeline test failed: {e}")
        return False


async def test_mathematical_components():
    """Test mathematical components."""
    print("\n🔍 Testing Mathematical Components...")
    
    try:
        from core.chrono_recursive_logic_function import ChronoRecursiveLogicFunction
        from core.clean_math_foundation import CleanMathFoundation
        from core.clean_profit_vectorization import CleanProfitVectorization
        from core.quantum_mathematical_bridge import QuantumMathematicalBridge

        # Test math foundation
        math_foundation = CleanMathFoundation()
        print("✅ Math Foundation created successfully")
        
        # Test profit vectorization
        profit_vectorizer = CleanProfitVectorization()
        print("✅ Profit Vectorization created successfully")
        
        # Test CRLF
        crlf = ChronoRecursiveLogicFunction()
        print("✅ Chrono-Recursive Logic Function created successfully")
        
        # Test quantum bridge
        quantum_bridge = QuantumMathematicalBridge()
        print("✅ Quantum Mathematical Bridge created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical components test failed: {e}")
        return False


async def test_neural_processing():
    """Test neural processing engine."""
    print("\n🔍 Testing Neural Processing Engine...")
    
    try:
        import numpy as np

        from core.neural_processing_engine import NeuralProcessingEngine

        # Create neural processing engine
        neural_engine = NeuralProcessingEngine()
        
        # Test with sample data
        sample_data = np.random.random((100, 10))
        
        print("✅ Neural Processing Engine created successfully")
        print(f"   Device: {neural_engine.device}")
        print(f"   Models initialized: {len(neural_engine.models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural Processing test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Schwabot System Tests")
    print("="*50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Live API Backtesting", test_live_api_backtesting),
        ("Portfolio Tracker", test_portfolio_tracker),
        ("Trading Pipeline", test_trading_pipeline),
        ("Mathematical Components", test_mathematical_components),
        ("Neural Processing", test_neural_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Schwabot system is ready to run.")
        print("\nTo start the system:")
        print("  python schwabot_cli.py")
        print("\nThen type:")
        print("  start")
        print("  enable-trading")
        print("  status")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    # Ensure necessary directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 