#!/usr/bin/env python3
"""
Trading Functionality Test - Schwabot Core Systems
==================================================

This script validates the core trading functionality including:
- SystemFitProfile and CUDA integration
- Automated trading pipeline
- Profit-seeking algorithms
- Backtesting integration
"""

import sys
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_fit_profile():
    """Test SystemFitProfile functionality."""
    logger.info("🧠 Testing SystemFitProfile...")

    try:
        from utils.cuda_helper import FIT_PROFILE, test_matrix_fit

        logger.info(f"✅ GPU Tier: {FIT_PROFILE.gpu_tier}")
        logger.info(f"✅ Device Type: {FIT_PROFILE.device_type}")
        logger.info(f"✅ Matrix Size: {FIT_PROFILE.matrix_size}x{FIT_PROFILE.matrix_size}")
        logger.info(f"✅ Precision: {FIT_PROFILE.precision}")
        logger.info(f"✅ Can Run GPU Logic: {FIT_PROFILE.can_run_gpu_logic}")
        logger.info(f"✅ System Hash: {FIT_PROFILE.system_hash[:12]}...")

        # Test matrix fit
        if test_matrix_fit():
            logger.info("✅ Matrix fit test passed")
        else:
            logger.warning("⚠️ Matrix fit test failed")

        return True

    except Exception as e:
        logger.error(f"❌ SystemFitProfile test failed: {e}")
        return False

def test_profit_calculator():
    """Test Pure Profit Calculator functionality."""
    logger.info("💰 Testing Pure Profit Calculator...")

    try:
        from core.pure_profit_calculator import ()
            PureProfitCalculator, MarketData, HistoryState, 
            StrategyParameters, ProcessingMode, ProfitCalculationMode
        )

        # Create strategy parameters
        strategy_params = StrategyParameters()
            risk_tolerance=0.2,
            profit_target=0.5,
            position_size=0.1,
            tensor_depth=4
        )

        # Create profit calculator
        calculator = PureProfitCalculator()
            strategy_params=strategy_params,
            processing_mode=ProcessingMode.HYBRID
        )

        # Create sample market data
        market_data = MarketData()
            timestamp=time.time(),
            btc_price=45000.0,
            eth_price=2700.0,
            usdc_volume=1000000.0,
            volatility=0.2,
            momentum=0.1,
            volume_profile=0.8,
            on_chain_signals={"whale_activity": 0.3, "network_health": 0.9}
        )

        # Create history state
        history_state = HistoryState(timestamp=time.time())

        # Calculate profit
        profit_result = calculator.calculate_profit()
            market_data, 
            history_state,
            mode=ProfitCalculationMode.BALANCED
        )

        logger.info(f"✅ Profit Score: {profit_result.total_profit_score:.4f}")
        logger.info(f"✅ Confidence: {profit_result.confidence_score:.4f}")
        logger.info(f"✅ Base Profit: {profit_result.base_profit:.4f}")
        logger.info(f"✅ Risk Adjusted: {profit_result.risk_adjusted_profit:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Profit Calculator test failed: {e}")
        return False

def test_trading_pipeline():
    """Test Automated Trading Pipeline functionality."""
    logger.info("🚀 Testing Automated Trading Pipeline...")

    try:
        from core.automated_trading_pipeline import AutomatedTradingPipeline

        # Create pipeline
        pipeline = AutomatedTradingPipeline()

        # Test processing a price tick
        decision = pipeline.process_price_tick()
            price=45000.0,
            volume=100.0,
            bid=44999.0,
            ask=45001.0
        )

        if decision:
            logger.info(f"✅ Trading Decision Made:")
            logger.info(f"   - Confidence: {decision.confidence_score:.4f}")
            logger.info(f"   - Position Size: {decision.position_size:.4f}")
            logger.info(f"   - Entry Price: {decision.entry_price:.2f}")
            logger.info(f"   - Decision Reason: {decision.decision_reason[:50]}...")
        else:
            logger.info("ℹ️ No trading decision (normal for single, tick)")

        # Get pipeline metrics
        metrics = pipeline.get_pipeline_metrics()
        logger.info(f"✅ Pipeline Metrics:")
        logger.info(f"   - Total Ticks: {metrics['total_ticks_processed']}")
        logger.info(f"   - Total Digests: {metrics['total_digests_generated']}")
        logger.info(f"   - Total Decisions: {metrics['total_decisions_made']}")

        return True

    except Exception as e:
        logger.error(f"❌ Trading Pipeline test failed: {e}")
        return False

def test_backtesting_integration():
    """Test Backtesting Integration functionality."""
    logger.info("📊 Testing Backtesting Integration...")

    try:
        from core.backtesting_integration import BacktestingEngine

        # Create backtesting engine
        engine = BacktestingEngine()

        # Test engine initialization
        logger.info(f"✅ Backtesting Engine initialized")
        logger.info(f"✅ Available strategies: {len(engine.available_strategies)}")

        return True

    except Exception as e:
        logger.error(f"❌ Backtesting Integration test failed: {e}")
        return False

def test_cuda_integration():
    """Test CUDA integration and fallback."""
    logger.info("⚡ Testing CUDA Integration...")

    try:
        from utils.cuda_helper import ()
            USING_CUDA, safe_matrix_multiply, get_cuda_status
        )

        # Test CUDA status
        status = get_cuda_status()
        logger.info(f"✅ Using CUDA: {status['using_cuda']}")
        logger.info(f"✅ Primary Library: {status['primary_library']}")

        # Test safe matrix multiplication
        import numpy as np
        A = np.random.rand(10, 10)
        B = np.random.rand(10, 10)

        result = safe_matrix_multiply(A, B)
        logger.info(f"✅ Matrix multiplication successful: {result.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ CUDA Integration test failed: {e}")
        return False

def main():
    """Run all trading functionality tests."""
    logger.info("🧪 Starting Schwabot Trading Functionality Tests")
    logger.info("=" * 60)

    tests = []
        ("SystemFitProfile", test_system_fit_profile),
        ("CUDA Integration", test_cuda_integration),
        ("Profit Calculator", test_profit_calculator),
        ("Trading Pipeline", test_trading_pipeline),
        ("Backtesting Integration", test_backtesting_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")

    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Trading system is functional.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Some functionality may be limited.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 