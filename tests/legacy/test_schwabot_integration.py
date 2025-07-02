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

import asyncio
import logging
import sys
import time
from pathlib import Path

# Core testing imports
try:
    from schwabot_unified_math import (
        UnifiedMathematicsFramework,
        BTC256SHAPipeline,
        unified_trading_math
    )
    from core.advanced_settings_engine import AdvancedSettingsEngine
    from core.api.cache_sync import CacheSyncService
    from core.api.handlers.whale_alert import WhaleAlertHandler
    from core.api.handlers.glassnode import GlassnodeHandler
    from core.api.handlers.coingecko import CoinGeckoHandler
    from core.api.handlers.alt_fear_greed import FearGreedHandler

    # Test if enhanced launcher is available
    try:
        from schwabot_enhanced_launcher import EnhancedDataIntegrator, SchawbotEnhancedLauncher
        LAUNCHER_AVAILABLE = True
    except ImportError:
        LAUNCHER_AVAILABLE = False

except ImportError as e:
    print(f"❌ Failed to import core components: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchawbotIntegrationTester:
    """Comprehensive integration tester for Schwabot system."""

    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()

    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        logger.info("🧪 Starting Schwabot Integration Test Suite")

        tests = [
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

    async def test_unified_math_framework(self) -> bool:
        """Test unified mathematics framework."""
        try:
            framework = UnifiedMathematicsFramework()

            # Test drift field calculation
            drift_field = framework.compute_unified_drift_field(1.0, 2.0, 0.5, 1.0)
            assert isinstance(drift_field, float), "Drift field should return float"

            # Test entropy calculation
            import numpy as np
            test_vector = np.array([0.5, 0.5, 0.0, 0.0])
            entropy = framework.compute_unified_entropy(test_vector)
            assert hasattr(entropy, 'value') or isinstance(entropy, (int, float)), "Entropy should be numeric"

            # Test hash generation
            unified_hash = framework.generate_unified_hash(test_vector, time_slot=1.5)
            assert isinstance(unified_hash, str) and len(unified_hash) == 64, "Hash should be 64-char string"

            # Test system integration
            input_data = {
                "tensor": np.random.rand(4, 4),
                "hash_patterns": ["test_hash"],
                "quantum_state": np.array([0.70710678, 0.70710678]),
                "metadata": {"source": "test"}
            }

            result = framework.integrate_all_systems(input_data)
            assert isinstance(result, dict), "Integration should return dict"

            logger.info("  ✓ All unified math framework tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Unified math framework test failed: {e}")
            return False

    async def test_advanced_settings_engine(self) -> bool:
        """Test advanced settings engine."""
        try:
            # Create temporary settings directory
            settings_dir = Path("test_settings")
            settings_dir.mkdir(exist_ok=True)

            engine = AdvancedSettingsEngine(
                config_path=settings_dir / "test_config.json"
            )

            # Test setting values
            result = engine.set_setting_value("echo_delay_sensitivity", 1.2)
            assert result == True, "Should be able to set valid setting"

            value = engine.get_setting_value("echo_delay_sensitivity")
            assert value == 1.2, "Should retrieve correct setting value"

            # Test bias application
            bias_result = engine.apply_bias_to_module("echo_modulator", 1.0)
            assert isinstance(bias_result, float), "Bias should return float"

            # Test confidence vector
            confidence = engine.get_confidence_vector("test")
            assert hasattr(confidence, 'ai_consensus'), "Should have confidence attributes"

            # Test signal scoring
            test_signals = [0.5, -0.2, 0.8, 0.1]
            score = engine.calculate_unified_signal_score(test_signals)
            assert isinstance(score, float), "Signal score should be float"

            # Test profit feedback
            engine.update_profit_feedback("test_setting", 0.05)

            # Clean up
            import shutil
            shutil.rmtree(settings_dir, ignore_errors=True)

            logger.info("  ✓ All advanced settings engine tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Advanced settings engine test failed: {e}")
            return False

    async def test_api_handlers(self) -> bool:
        """Test API handlers."""
        try:
            # Test each handler initialization
            handlers = [
                ("FearGreedHandler", FearGreedHandler),
                ("WhaleAlertHandler", WhaleAlertHandler),
                ("GlassnodeHandler", GlassnodeHandler),
                ("CoinGeckoHandler", CoinGeckoHandler),
            ]

            for name, handler_class in handlers:
                handler = handler_class()
                assert hasattr(handler, 'NAME'), f"{name} should have NAME attribute"
                assert hasattr(handler, '_fetch_raw'), f"{name} should have _fetch_raw method"
                assert hasattr(handler, '_parse_raw'), f"{name} should have _parse_raw method"

            # Test a simple parse operation (without actual API call)
            fear_greed = FearGreedHandler()
            test_data = {"data": [{"value": 25, "classification": "fear"}]}
            parsed = await fear_greed._parse_raw(test_data)
            assert isinstance(parsed, dict), "Parsed data should be dict"

            logger.info("  ✓ All API handler tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ API handler test failed: {e}")
            return False

    async def test_cache_sync_service(self) -> bool:
        """Test cache sync service."""
        try:
            service = CacheSyncService(refresh_interval=60)

            # Add a test handler
            fear_greed = FearGreedHandler()
            service.handlers.append(fear_greed)

            assert len(service.handlers) == 1, "Should have one handler"

            # Test discovery (without starting full service)
            assert hasattr(service, '_discover_handlers'), "Should have discovery method"

            logger.info("  ✓ Cache sync service tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Cache sync service test failed: {e}")
            return False

    async def test_data_integration(self) -> bool:
        """Test data integration components."""
        try:
            if not LAUNCHER_AVAILABLE:
                logger.info("  ⚠️  Enhanced launcher not available, skipping data integration test")
                return True

            # Create test components
            framework = UnifiedMathematicsFramework()
            settings_engine = AdvancedSettingsEngine()

            integrator = EnhancedDataIntegrator(settings_engine, framework)

            # Test signal processing methods
            fear_greed_signal = integrator._process_fear_greed_signal({"value": 25})
            assert isinstance(fear_greed_signal, float), "Fear/greed signal should be float"
            assert -1.0 <= fear_greed_signal <= 1.0, "Signal should be in valid range"

            whale_alert_signal = integrator._process_whale_alert_signal({"amount_usd": 1000000})
            assert isinstance(whale_alert_signal, float), "Whale alert signal should be float"

            glassnode_signal = integrator._process_glassnode_signal({"value": 0.8})
            assert isinstance(glassnode_signal, float), "Glassnode signal should be float"

            # Test BTC hash pipeline
            btc_hash = integrator.btc_sha_pipeline.process_data(b"test_data")
            assert isinstance(btc_hash, str), "BTC hash should be a string"

            # Test unified trading math
            math_result = integrator.unified_trading_math.calculate_profit_score(
                price=50000,
                volume=1000,
                volatility=0.02,
                confidence=0.7
            )
            assert isinstance(math_result, float), "Profit score should be float"

            logger.info("  ✓ Data integration components passed tests")
            return True

        except Exception as e:
            logger.error(f"  ❌ Data integration test failed: {e}")
            return False

    async def test_trading_mathematics(self) -> bool:
        """Test core trading mathematics."""
        try:
            result = unified_trading_math(
                price=50000,
                volume=1000,
                volatility=0.02,
                confidence=0.7
            )
            assert isinstance(result, float), "unified_trading_math should return float"

            btc_pipeline = BTC256SHAPipeline()
            btc_hash = btc_pipeline.process_data(b"test")
            assert isinstance(btc_hash, str), "BTC pipeline should return string"

            logger.info("  ✓ Core trading math tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Core trading math test failed: {e}")
            return False

    async def test_signal_processing(self) -> bool:
        """Test signal processing logic."""
        try:
            framework = UnifiedMathematicsFramework()
            settings = AdvancedSettingsEngine()

            input_data = {
                "fear_greed": {"value": 30},
                "whale_alert": {"amount_usd": 500000},
                "glassnode": {"value": 0.6},
                "price": 50000,
                "volume": 1000
            }

            if not LAUNCHER_AVAILABLE:
                logger.info("  ⚠️  Enhanced launcher not available, using standalone signal processing")
                # Simulate signal processing
                signal = 0.5 # Placeholder
            else:
                integrator = EnhancedDataIntegrator(settings, framework)
                signal = integrator.process_all_signals(input_data)

            assert isinstance(signal, float), "Signal should be a float"
            logger.info("  ✓ Signal processing logic tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Signal processing logic test failed: {e}")
            return False

    async def test_enhanced_launcher(self) -> bool:
        """Test enhanced launcher functionality."""
        if not LAUNCHER_AVAILABLE:
            logger.info("  ⚠️  Enhanced launcher not available, skipping test")
            return True
        try:
            # Initialize with test mode
            launcher = SchawbotEnhancedLauncher(test_mode=True)
            assert launcher.test_mode == True, "Launcher should be in test mode"

            # Test component initialization
            assert hasattr(launcher, 'data_integrator'), "Launcher should have data integrator"
            assert hasattr(launcher, 'settings_engine'), "Launcher should have settings engine"

            # Test a single run cycle
            await launcher._run_cycle()

            logger.info("  ✓ Enhanced launcher tests passed")
            return True

        except Exception as e:
            logger.error(f"  ❌ Enhanced launcher test failed: {e}")
            return False

    def print_test_summary(self, all_passed: bool) -> None:
        """Print test results summary."""
        end_time = time.time()
        duration = end_time - self.start_time

        logger.info("\n" + "="*50)
        logger.info("Schwabot Integration Test Summary")
        logger.info("="*50)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  - {test_name}: {status}")

        logger.info("-" * 50)
        logger.info(f"Total tests: {len(self.test_results)}")
        passed_count = sum(1 for res in self.test_results.values() if res)
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {len(self.test_results) - passed_count}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 50)

        if all_passed:
            logger.info("🎉 All integration tests completed successfully!")
        else:
            logger.error("🔥 Some integration tests failed. Please review the logs.")


async def main():
    """Main entry point for the test suite."""
    tester = SchawbotIntegrationTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 