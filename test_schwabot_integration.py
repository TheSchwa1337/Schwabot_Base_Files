import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from core.advanced_settings_engine import AdvancedSettingsEngine
from core.api.cache_sync import CacheSyncService
from core.api.handlers.alt_fear_greed import FearGreedHandler
from core.api.handlers.coingecko import CoinGeckoHandler
from core.api.handlers.glassnode import GlassnodeHandler
from core.api.handlers.whale_alert import WhaleAlertHandler

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Integration Test Suite
===============================

Comprehensive test script to verify all components of the Schwabot integration
work together properly. This tests:

- Unified Mathematics Framework
- Advanced Settings Engine
- API Handlers and Cache Sync
- Data Integration Pipeline
- Trading Signal Processing
"""


# Core testing imports
    try:
        UnifiedMathematicsFramework,
        BTC256SHAPipeline,
        unified_trading_math,
    )

    # Test if enhanced launcher is available
    try:
            EnhancedDataIntegrator,
            SchawbotEnhancedLauncher,
        )

        LAUNCHER_AVAILABLE = True
    except ImportError:
        LAUNCHER_AVAILABLE = False

except ImportError as e:
    print(f"❌ Failed to import core components: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchawbotIntegrationTester:
    """Comprehensive integration tester for Schwabot system."""

    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()

    async def run_all_tests():-> bool:
        """Run all integration tests."""
        logger.info("🧪 Starting Schwabot Integration Test Suite")

        tests = []
            ("Unified Mathematics Framework", self.test_unified_math_framework),
            ("Advanced Settings Engine", self.test_advanced_settings_engine),
            ("API Handlers", self.test_api_handlers),
            ("Cache Sync Service", self.test_cache_sync_service),
            ("Data Integration", self.test_data_integration),
            ("Trading Mathematics", self.test_trading_mathematics),
            ("Signal Processing", self.test_signal_processing),
        ]

        if LAUNCHER_AVAILABLE:
            tests.append(("Enhanced Launcher", self.test_enhanced_launcher))

        all_passed = True

        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = False
                all_passed = False

        # Print summary
        self.print_test_summary(all_passed)
        return all_passed

    async def test_unified_math_framework():-> bool:
        """Test unified mathematics framework."""
        try:
            framework = UnifiedMathematicsFramework()

            # Test drift field calculation
            drift_field = framework.compute_unified_drift_field(1.0, 2.0, 0.5, 1.0)
            assert isinstance(drift_field, float), "Drift field should return float"

            # Test entropy calculation

            test_vector = np.array([0.5, 0.5, 0.0, 0.0])
            entropy = framework.compute_unified_entropy(test_vector)
            assert hasattr(entropy, "value") or isinstance(entropy, (int, float)), ()
                "Entropy should be numeric"
            )

            # Test hash generation
            unified_hash = framework.generate_unified_hash(test_vector, time_slot=1.5)
            assert isinstance(unified_hash, str) and len(unified_hash) == 64, ()
                "Hash should be 64-char string"
            )

            # Test system integration
            input_data = {}
                "tensor": np.random.rand(4, 4),
                "hash_patterns": ["test_hash"],
                "quantum_state": np.array([0.70710678, 0.70710678]),
                "metadata": {"source": "test"},
            }

            result = framework.integrate_all_systems(input_data)
            assert isinstance(result, dict), "Integration should return dict"

            logger.info("  ✓ All unified math framework tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Unified math framework test failed: {e}")
            return False

    async def test_advanced_settings_engine():-> bool:
        """Test advanced settings engine."""
        try:
            # Create temporary settings directory
            settings_dir = Path("test_settings")
            settings_dir.mkdir(exist_ok=True)

            engine = AdvancedSettingsEngine()
                config_path=settings_dir / "test_config.json"
            )

            # Test setting values
            result = engine.set_setting_value("echo_delay_sensitivity", 1.2)
            assert result, "Should be able to set valid setting"

            value = engine.get_setting_value("echo_delay_sensitivity")
            assert value == 1.2, "Should retrieve correct setting value"

            # Test bias application
            bias_result = engine.apply_bias_to_module("echo_modulator", 1.0)
            assert isinstance(bias_result, float), "Bias should return float"

            # Test confidence vector
            confidence = engine.get_confidence_vector("test")
            assert hasattr(confidence, "ai_consensus"), ()
                "Should have confidence attributes"
            )

            # Test signal scoring
            test_signals = [0.5, -0.2, 0.8, 0.1]
            score = engine.calculate_unified_signal_score(test_signals)
            assert isinstance(score, float), "Signal score should be float"

            # Test profit feedback
            engine.update_profit_feedback("test_setting", 0.5)

            # Clean up

            shutil.rmtree(settings_dir, ignore_errors=True)

            logger.info("  ✓ All advanced settings engine tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Advanced settings engine test failed: {e}")
            return False

    async def test_api_handlers():-> bool:
        """Test API handlers."""
        try:
            # Test each handler initialization
            handlers = []
                ("FearGreedHandler", FearGreedHandler),
                ("WhaleAlertHandler", WhaleAlertHandler),
                ("GlassnodeHandler", GlassnodeHandler),
                ("CoinGeckoHandler", CoinGeckoHandler),
            ]

            for name, handler_class in handlers:
                handler = handler_class()
                assert hasattr(handler, "NAME"), f"{name} should have NAME attribute"
                assert hasattr(handler, "_fetch_raw"), ()
                    f"{name} should have _fetch_raw method"
                )
                assert hasattr(handler, "_parse_raw"), ()
                    f"{name} should have _parse_raw method"
                )

            # Test a simple parse operation (without actual API, call)
            fear_greed = FearGreedHandler()
            test_data = {"data": [{"value": 25, "classification": "fear"}]}
            parsed = await fear_greed._parse_raw(test_data)
            assert isinstance(parsed, dict), "Parsed data should be dict"

            logger.info("  ✓ All API handler tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ API handler test failed: {e}")
            return False

    async def test_cache_sync_service():-> bool:
        """Test cache sync service."""
        try:
            service = CacheSyncService(refresh_interval=60)

            # Add a test handler
            fear_greed = FearGreedHandler()
            service.handlers.append(fear_greed)

            assert len(service.handlers) == 1, "Should have one handler"

            # Test discovery (without starting full, service)
            assert hasattr(service, "_discover_handlers"), ()
                "Should have discovery method"
            )

            logger.info("  ✓ Cache sync service tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Cache sync service test failed: {e}")
            return False

    async def test_data_integration():-> bool:
        """Test data integration components."""
        try:
            if not LAUNCHER_AVAILABLE:
                logger.info()
                    "  ⚠️  Enhanced launcher not available, skipping data integration test"
                )
                return True

            # Create test components
            framework = UnifiedMathematicsFramework()
            settings_engine = AdvancedSettingsEngine()

            integrator = EnhancedDataIntegrator(settings_engine, framework)

            # Test signal processing methods
            fear_greed_signal = integrator._process_fear_greed_signal({"value": 25})
            assert isinstance(fear_greed_signal, float), ()
                "Fear/greed signal should be float"
            )
            assert -1.0 <= fear_greed_signal <= 1.0, "Signal should be in valid range"

            whale_signal = integrator._process_whale_signal()
                {"summary": {"whale_activity_score": 60, "total_volume_usd": 1000000}}
            )
            assert isinstance(whale_signal, float), "Whale signal should be float"

            logger.info("  ✓ Data integration tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Data integration test failed: {e}")
            return False

    async def test_trading_mathematics():-> bool:
        """Test trading mathematics components."""
        try:
            # Test profit optimization
            profit_score = unified_trading_math.calculate_profit_optimization()
                50000.0, 1000.0, "BTC"
            )
            assert isinstance(profit_score, float), "Profit score should be float"
            assert profit_score >= 0, "Profit score should be non-negative"

            # Test risk-adjusted return
            test_returns = [0.5, -0.2, 0.3, 0.1, -0.1]
            sharpe_ratio = unified_trading_math.calculate_risk_adjusted_return()
                test_returns
            )
            assert isinstance(sharpe_ratio, float), "Sharpe ratio should be float"

            # Test portfolio optimization

            weights = np.array([0.6, 0.4])
            returns = np.array([[0.5, 0.2], [-0.2, 0.3], [0.1, -0.1]])
            portfolio_metrics = unified_trading_math.calculate_portfolio_optimization()
                weights, returns
            )
            assert isinstance(portfolio_metrics, dict), ()
                "Portfolio metrics should be dict"
            )
            assert "portfolio_return" in portfolio_metrics, ()
                "Should have portfolio return"
            )

            logger.info("  ✓ Trading mathematics tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Trading mathematics test failed: {e}")
            return False

    async def test_signal_processing():-> bool:
        """Test signal processing pipeline."""
        try:
            framework = UnifiedMathematicsFramework()

            # Test BTC pipeline
            btc_pipeline = BTC256SHAPipeline(framework)
            result = btc_pipeline.process_price_data(50000.0, time.time())

            assert isinstance(result, dict), "Pipeline result should be dict"
            assert "price" in result, "Should contain price"
            assert "hash" in result, "Should contain hash"
            assert "entropy" in result, "Should contain entropy"

            # Test pipeline status
            status = btc_pipeline.get_pipeline_status()
            assert isinstance(status, dict), "Status should be dict"

            logger.info("  ✓ Signal processing tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Signal processing test failed: {e}")
            return False

    async def test_enhanced_launcher():-> bool:
        """Test enhanced launcher components."""
        try:
            if not LAUNCHER_AVAILABLE:
                logger.info("  ⚠️  Enhanced launcher not available, skipping test")
                return True

            # Test launcher initialization (without, starting)
            launcher = SchawbotEnhancedLauncher()

            assert hasattr(launcher, "initialize"), "Should have initialize method"
            assert hasattr(launcher, "start"), "Should have start method"
            assert hasattr(launcher, "shutdown"), "Should have shutdown method"

            # Test component initialization (without full, startup)
            assert launcher.performance_metrics is not None, ()
                "Should have performance metrics"
            )
            assert launcher.tasks is not None, "Should have tasks dict"

            logger.info("  ✓ Enhanced launcher tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Enhanced launcher test failed: {e}")
            return False

    def print_test_summary():-> None:
        """Print comprehensive test summary."""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("🧪 SCHWABOT INTEGRATION TEST SUMMARY")
        print("=" * 60)

        passed_count = sum(1 for result in self.test_results.values() if result)
        total_count = len(self.test_results)

        print(f"Total Tests: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Runtime: {runtime:.2f} seconds")
        print()

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name}")

        print()
        if all_passed:
            print("🎉 ALL TESTS PASSED! Schwabot integration is working correctly.")
            print("🚀 System is ready for enhanced trading operations.")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
            print("🔧 Fix the issues before running the full system.")

        print("=" * 60)


async def main():
    """Main test runner."""
    tester = SchawbotIntegrationTester()
    success = await tester.run_all_tests()

    if success:
        print("\n🌟 Integration test completed successfully!")
        print("You can now run the enhanced launcher with:")
        print("   python schwabot_enhanced_launcher.py")
    else:
        print("\n❌ Integration test failed!")
        print("Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)
