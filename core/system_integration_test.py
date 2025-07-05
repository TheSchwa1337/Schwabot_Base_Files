#!/usr/bin/env python3
"""
System Integration Test for Schwabot Trading System
Validates all core components and their interactions.
"""

import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_antipole_router():
    """Test the antipole router system."""
    try:
        from .antipole_router import (
            AntipoleRouter, StrategyVector, TradeMemory,
            ProfitFadeDetectionEngine, HashEchoPolarityVerifier,
            StrategyInversionVectorizer, MemoryMirrorAllocator,
            FractalDriftCorrector, CPUGPUDispatchScheduler,
            RegistryUpdateBasketReweigher
        )
        
        logger.info("✅ Antipole Router imports successful")
        
        # Test individual components
        test_results = {}
        
        # Test PFDE
        try:
            pfde = ProfitFadeDetectionEngine()
            pfde.update_profit(100.0)
            pfde.update_profit(90.0)
            pfde.update_profit(80.0)
            fade_detected = pfde.detect_fade()
            test_results['PFDE'] = 'PASS'
            logger.info("✅ PFDE test passed")
        except Exception as e:
            test_results['PFDE'] = f'FAIL: {e}'
            logger.error(f"❌ PFDE test failed: {e}")
        
        # Test HEPV
        try:
            hepv = HashEchoPolarityVerifier()
            strategy = StrategyVector(asset="BTC", risk_profile="aggressive")
            antipole = StrategyVector(asset="USDC", risk_profile="conservative")
            hash_valid = hepv.verify_antipole_hash(strategy, antipole)
            test_results['HEPV'] = 'PASS'
            logger.info("✅ HEPV test passed")
        except Exception as e:
            test_results['HEPV'] = f'FAIL: {e}'
            logger.error(f"❌ HEPV test failed: {e}")
        
        # Test SIV
        try:
            siv = StrategyInversionVectorizer()
            strategy = StrategyVector(asset="BTC", risk_profile="aggressive")
            antipole = siv.invert_strategy(strategy)
            assert antipole.asset == "USDC"
            assert antipole.risk_profile == "conservative"
            test_results['SIV'] = 'PASS'
            logger.info("✅ SIV test passed")
        except Exception as e:
            test_results['SIV'] = f'FAIL: {e}'
            logger.error(f"❌ SIV test failed: {e}")
        
        # Test MMA
        try:
            mma = MemoryMirrorAllocator()
            memory = TradeMemory()
            memory.add_entry(50000.0, 0.1)
            memory.add_entry(51000.0, 0.2)
            mirrored = mma.mirror_memory(memory)
            assert len(mirrored.entries) == 2
            test_results['MMA'] = 'PASS'
            logger.info("✅ MMA test passed")
        except Exception as e:
            test_results['MMA'] = f'FAIL: {e}'
            logger.error(f"❌ MMA test failed: {e}")
        
        # Test FDC
        try:
            fdc = FractalDriftCorrector()
            fdc.update_price_history(50000.0, datetime.now())
            fdc.update_price_history(51000.0, datetime.now())
            fdc.update_price_history(52000.0, datetime.now())
            drift = fdc.calculate_drift_function(datetime.now())
            test_results['FDC'] = 'PASS'
            logger.info("✅ FDC test passed")
        except Exception as e:
            test_results['FDC'] = f'FAIL: {e}'
            logger.error(f"❌ FDC test failed: {e}")
        
        # Test full router
        try:
            router = AntipoleRouter()
            strategy = StrategyVector(asset="BTC", risk_profile="aggressive")
            memory = TradeMemory()
            memory.add_entry(50000.0, 0.1)
            
            # Simulate profit fade
            profit_data = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
            current_values = {"BTC": 50000.0, "ETH": 3000.0, "XRP": 0.5, "USDC": 1.0}
            
            result = router.antipole_router(strategy, profit_data, memory, current_values)
            antipole_state = router.get_antipole_state()
            
            test_results['AntipoleRouter'] = 'PASS'
            logger.info("✅ Full Antipole Router test passed")
        except Exception as e:
            test_results['AntipoleRouter'] = f'FAIL: {e}'
            logger.error(f"❌ Full Antipole Router test failed: {e}")
        
        return test_results
        
    except ImportError as e:
        logger.error(f"❌ Import error in antipole router test: {e}")
        return {'Import': f'FAIL: {e}'}
    except Exception as e:
        logger.error(f"❌ Unexpected error in antipole router test: {e}")
        return {'Unexpected': f'FAIL: {e}'}

def test_automated_trading_engine():
    """Test the automated trading engine."""
    try:
        from .automated_trading_engine import (
            AutomatedTradingEngine, TradingSignal, BatchOrder,
            ExchangeManager, PriceTracker, OrderManager, BatchOrderProcessor
        )
        
        logger.info("✅ Automated Trading Engine imports successful")
        
        # Test components
        test_results = {}
        
        # Test TradingSignal
        try:
            signal = TradingSignal(
                symbol="BTC/USD",
                side="buy",
                quantity=0.1,
                price=50000.0
            )
            assert signal.symbol == "BTC/USD"
            test_results['TradingSignal'] = 'PASS'
            logger.info("✅ TradingSignal test passed")
        except Exception as e:
            test_results['TradingSignal'] = f'FAIL: {e}'
            logger.error(f"❌ TradingSignal test failed: {e}")
        
        # Test BatchOrder
        try:
            batch = BatchOrder(
                symbol="BTC/USD",
                side="buy",
                total_quantity=1.0,
                batch_count=10,
                price_range=(49000.0, 51000.0),
                spread_seconds=60,
                strategy="test"
            )
            assert batch.batch_count == 10
            test_results['BatchOrder'] = 'PASS'
            logger.info("✅ BatchOrder test passed")
        except Exception as e:
            test_results['BatchOrder'] = f'FAIL: {e}'
            logger.error(f"❌ BatchOrder test failed: {e}")
        
        return test_results
        
    except ImportError as e:
        logger.error(f"❌ Import error in trading engine test: {e}")
        return {'Import': f'FAIL: {e}'}
    except Exception as e:
        logger.error(f"❌ Unexpected error in trading engine test: {e}")
        return {'Unexpected': f'FAIL: {e}'}

def test_core_dependencies():
    """Test core system dependencies."""
    test_results = {}
    
    # Test numpy
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert len(arr) == 3
        test_results['numpy'] = 'PASS'
        logger.info("✅ NumPy test passed")
    except Exception as e:
        test_results['numpy'] = f'FAIL: {e}'
        logger.error(f"❌ NumPy test failed: {e}")
    
    # Test logging
    try:
        import logging
        test_logger = logging.getLogger('test')
        test_logger.info("Test message")
        test_results['logging'] = 'PASS'
        logger.info("✅ Logging test passed")
    except Exception as e:
        test_results['logging'] = f'FAIL: {e}'
        logger.error(f"❌ Logging test failed: {e}")
    
    # Test threading
    try:
        import threading
        lock = threading.Lock()
        with lock:
            pass
        test_results['threading'] = 'PASS'
        logger.info("✅ Threading test passed")
    except Exception as e:
        test_results['threading'] = f'FAIL: {e}'
        logger.error(f"❌ Threading test failed: {e}")
    
    return test_results

def run_comprehensive_system_test():
    """Run comprehensive system test."""
    logger.info("🚀 Starting Comprehensive System Test")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Test core dependencies
    logger.info("Testing core dependencies...")
    all_results['Dependencies'] = test_core_dependencies()
    
    # Test antipole router
    logger.info("Testing antipole router...")
    all_results['AntipoleRouter'] = test_antipole_router()
    
    # Test automated trading engine
    logger.info("Testing automated trading engine...")
    all_results['TradingEngine'] = test_automated_trading_engine()
    
    # Generate report
    logger.info("=" * 60)
    logger.info("📊 SYSTEM TEST REPORT")
    logger.info("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        logger.info(f"\n{category}:")
        for test_name, result in results.items():
            total_tests += 1
            if result == 'PASS':
                passed_tests += 1
                logger.info(f"  ✅ {test_name}: {result}")
            else:
                logger.info(f"  ❌ {test_name}: {result}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("=" * 60)
    logger.info(f"📈 OVERALL RESULTS:")
    logger.info(f"   Total Tests: {total_tests}")
    logger.info(f"   Passed: {passed_tests}")
    logger.info(f"   Failed: {total_tests - passed_tests}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        logger.info("🎉 SYSTEM STATUS: HEALTHY")
    elif success_rate >= 60:
        logger.info("⚠️  SYSTEM STATUS: NEEDS ATTENTION")
    else:
        logger.info("🚨 SYSTEM STATUS: CRITICAL ISSUES")
    
    logger.info("=" * 60)
    
    return all_results, success_rate

if __name__ == "__main__":
    try:
        results, success_rate = run_comprehensive_system_test()
        
        if success_rate >= 80:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        logger.error(f"💥 Critical error during system test: {e}")
        logger.error(traceback.format_exc())
        sys.exit(2)  # Critical failure 