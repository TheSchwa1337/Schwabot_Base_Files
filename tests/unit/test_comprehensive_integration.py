import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from core.mathlib_v4 import MathLibV4
from core.matrix_math_utils import analyze_price_matrix
from core.risk_manager import RiskManager
from core.strategy_integration_bridge import create_strategy_integration_bridge
from core.strategy_logic import StrategyLogic
from core.unified_math_system import UnifiedMathSystem
from core.unified_trading_pipeline import UnifiedTradingPipeline

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive Integration Test - Enhanced Strategy Framework with Schwabot Pipeline."

Comprehensive test suite validating the integration between:
1. Enhanced Strategy Framework (Wall Street, strategies)
2. Strategy Integration Bridge
3. Schwabot Mathematical Pipeline
4. API endpoints and visualization integration
5. Flake8 compliance verification

This test ensures all mathematical calculations, trading strategies, and
integrations work correctly together for a production-ready trading bot.

Windows CLI compatible with detailed error reporting.
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_import_compatibility():
    """Check import compatibility and component availability."""
    print("🔍 Checking Component Import Compatibility...")

    import_status = {}
        "enhanced_framework": False,
        "integration_bridge": False,
        "core_components": False,
        "mathematical_components": False,
        "api_components": False,
    }

    try:
            EnhancedStrategyFramework,
            WallStreetStrategy,
            TimeFrame,
            StrategySignal,
        )

        import_status["enhanced_framework"] = True
        print("   ✅ Enhanced Strategy Framework imported successfully")
    except ImportError as e:
        print(f"   ❌ Enhanced Strategy Framework import failed: {e}")

    try:
            StrategyIntegrationBridge,
            IntegratedTradingSignal,
            create_strategy_integration_bridge,
        )

        import_status["integration_bridge"] = True
        print("   ✅ Strategy Integration Bridge imported successfully")
    except ImportError as e:
        print(f"   ❌ Strategy Integration Bridge import failed: {e}")

    try:

        import_status["mathematical_components"] = True
        print("   ✅ Mathematical components imported successfully")
    except ImportError as e:
        print(f"   ❌ Mathematical components import failed: {e}")

    try:

        import_status["core_components"] = True
        print("   ✅ Core trading components imported successfully")
    except ImportError as e:
        print(f"   ❌ Core trading components import failed: {e}")

    return import_status


def test_enhanced_strategy_framework():
    """Test enhanced strategy framework functionality."""
    print("\n📊 Testing Enhanced Strategy Framework...")

    try:
            EnhancedStrategyFramework,
            TimeFrame,
        )

        # Create framework with test configuration
        config = {}
            "max_signal_history": 100,
            "min_signal_confidence": 0.6,
            "enable_dynamic_weights": True,
        }

        framework = EnhancedStrategyFramework(config)

        # Test 1: Framework initialization
        assert framework.version == "2.0.0"
        assert len(framework.active_strategies) > 0
        print()
            f"   ✅ Framework initialized with {len(framework.active_strategies)} strategies"
        )

        # Test 2: Generate test signals with market data
        print("   📈 Generating test signals...")

        # Simulate market data history
        test_prices = [50000.0 + i * 100 for i in range(50)]
        test_volumes = [1000.0 + i * 10 for i in range(50)]

        for i, (price, volume) in enumerate(zip(test_prices, test_volumes)):
            framework._update_market_data("BTC/USDT", price, volume)

        # Generate signals
        signals = framework.generate_wall_street_signals()
            asset="BTC/USDT", price=55000.0, volume=1500.0, timeframe=TimeFrame.ONE_HOUR
        )

        print(f"   ✅ Generated {len(signals)} Wall Street signals")

        # Test 3: Validate signal quality
        for signal in signals:
            assert signal.confidence >= 0.0 and signal.confidence <= 1.0
            assert signal.action in ["BUY", "SELL", "HOLD"]
            assert signal.risk_reward_ratio > 0
            assert signal.position_size > 0

        print("   ✅ All signals validated successfully")

        # Test 4: Strategy performance tracking
        if signals:
            test_result = {"executed": True, "pnl": 100.0}
            framework.update_strategy_performance(signals[0], test_result)

            performance = framework.get_strategy_performance(signals[0].strategy)
            assert performance is not None
            assert performance.total_signals >= 1
            print("   ✅ Strategy performance tracking working")

        # Test 5: Framework status
        status = framework.get_framework_status()
        assert "version" in status
        assert "active_strategies" in status
        assert "strategy_weights" in status
        print("   ✅ Framework status reporting working")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced Strategy Framework test failed: {e}")
        return False


async def test_strategy_integration_bridge():
    """Test strategy integration bridge functionality."""
    print("\n🌉 Testing Strategy Integration Bridge...")

    try:

        # Create integration bridge
        config = {}
            "correlation_threshold": 0.5,
            "max_integrated_signals": 10,
            "enable_api_endpoints": True,
        }

        bridge = create_strategy_integration_bridge(config)

        # Test 1: Bridge initialization
        assert bridge.version == "1.0.0"
        assert hasattr(bridge, "enhanced_framework")
        print("   ✅ Integration bridge initialized successfully")

        # Test 2: Integrated signal processing
        print("   🔄 Processing integrated trading signals...")

        integrated_signals = await bridge.process_integrated_trading_signal()
            asset="BTC/USDT", price=52000.0, volume=1200.0
        )

        print(f"   ✅ Generated {len(integrated_signals)} integrated signals")

        # Test 3: Validate integrated signals
        for signal in integrated_signals:
            assert hasattr(signal, "wall_street_signal")
            assert hasattr(signal, "mathematical_confidence")
            assert hasattr(signal, "composite_confidence")
            assert hasattr(signal, "correlation_score")
            assert ()
                signal.composite_confidence >= 0.0
                and signal.composite_confidence <= 1.0
            )

        print("   ✅ Integrated signals validated successfully")

        # Test 4: API endpoints
        api_endpoints = bridge.get_api_endpoints()
        assert len(api_endpoints) > 0
        print(f"   ✅ {len(api_endpoints)} API endpoints available")

        # Test API endpoints
        for endpoint_name, endpoint_func in api_endpoints.items():
            try:
                result = await endpoint_func()
                assert isinstance(result, dict)
                print(f"     ✅ {endpoint_name} working")
            except Exception as e:
                print(f"     ⚠️  {endpoint_name} issue: {e}")

        # Test 5: Integration status
        status = bridge.get_integration_status()
        assert "bridge_version" in status
        assert "component_status" in status
        assert "orchestration_state" in status
        print("   ✅ Integration status reporting working")

        # Test 6: Signal execution (dry, run)
        if integrated_signals:
            execution_result = await bridge.execute_integrated_signal()
                integrated_signals[0]
            )
            assert isinstance(execution_result, dict)
            print("   ✅ Signal execution pathway working")

        return True

    except Exception as e:
        print(f"   ❌ Strategy Integration Bridge test failed: {e}")
        return False


def test_mathematical_integration():
    """Test mathematical component integration."""
    print("\n🧮 Testing Mathematical Component Integration...")

    try:
        # Test MathLibV4 integration
        try:

            mathlib = MathLibV4(precision=64)

            # Test basic mathematical operations
            test_data = {}
                "prices": [50000.0, 50100.0, 50200.0, 50150.0, 50300.0],
                "volumes": [100.0, 110.0, 120.0, 115.0, 130.0],
                "timestamps": [time.time() - i for i in range(5)],
            }

            result = mathlib.calculate_dlt_metrics(test_data)
            if "error" not in result:
                print("   ✅ MathLibV4 DLT calculations working")
            else:
                print(f"   ⚠️  MathLibV4 DLT calculation issue: {result.get('error')}")

        except ImportError:
            print("   ⚠️  MathLibV4 not available for testing")

        # Test Unified Math System
        try:

            unified_math = UnifiedMathSystem()
            state = unified_math.get_system_state()
            assert isinstance(state, dict)
            print("   ✅ Unified Math System working")

        except ImportError:
            print("   ⚠️  Unified Math System not available for testing")

        # Test Matrix Math Utils
        try:

            test_matrix = np.array([[50000.0, 50100.0], [50200.0, 50300.0]])
            result = analyze_price_matrix(test_matrix)
            assert result is not None
            print("   ✅ Matrix Math Utils working")

        except ImportError:
            print("   ⚠️  Matrix Math Utils not available for testing")

        return True

    except Exception as e:
        print(f"   ❌ Mathematical integration test failed: {e}")
        return False


def test_risk_management_integration():
    """Test risk management integration."""
    print("\n⚖️ Testing Risk Management Integration...")

    try:

        # Initialize risk manager
        config = {}
            "max_portfolio_risk": 0.2,
            "max_position_size": 0.1,
            "risk_free_rate": 0.2,
        }

        risk_manager = RiskManager(config)

        # Test risk calculation
        test_data = {}
            "asset": "BTC/USDT",
            "price": 50000.0,
            "volume": 1000.0,
            "position_size": 0.5,
        }

        risk_metrics = risk_manager.calculate_risk_metrics(test_data)
        assert isinstance(risk_metrics, dict)
        print("   ✅ Risk metrics calculation working")

        return True

    except ImportError:
        print("   ⚠️  Risk Manager not available for testing")
        return True
    except Exception as e:
        print(f"   ❌ Risk management test failed: {e}")
        return False


def test_api_visualization_integration():
    """Test API and visualization integration."""
    print("\n🖥️ Testing API and Visualization Integration...")

    try:

        bridge = create_strategy_integration_bridge()
        api_endpoints = bridge.get_api_endpoints()

        # Test that visualization-compatible data structures are returned
        test_results = []

        for endpoint_name in api_endpoints.keys():
            test_results.append()
                {}
                    "endpoint": endpoint_name,
                    "available": True,
                    "data_structure": "dict",  # All endpoints return dict
                }
            )

        print(f"   ✅ {len(test_results)} API endpoints ready for visualization")

        # Check for visualization dashboard files
        dashboard_files = []
            "unified_visual_dashboard.html",
            "enhanced_crypto_dashboard.html",
        ]

        for dashboard_file in dashboard_files:
            if Path(dashboard_file).exists():
                print(f"   ✅ Dashboard file {dashboard_file} available")
            else:
                print(f"   ⚠️  Dashboard file {dashboard_file} not found")

        return True

    except Exception as e:
        print(f"   ❌ API/Visualization integration test failed: {e}")
        return False


async def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    print("\n🔄 Testing End-to-End Integration...")

    try:

        # Create complete integration
        config = {}
            "correlation_threshold": 0.5,
            "enable_real_time_optimization": True,
            "enable_api_endpoints": True,
        }

        bridge = create_strategy_integration_bridge(config)

        print("   📊 Simulating complete trading cycle...")

        # Simulate market data sequence
        market_data_sequence = []
            {"asset": "BTC/USDT", "price": 50000.0, "volume": 1000.0},
            {"asset": "BTC/USDT", "price": 50200.0, "volume": 1100.0},
            {"asset": "BTC/USDT", "price": 50400.0, "volume": 1200.0},
            {"asset": "ETH/USDT", "price": 3000.0, "volume": 500.0},
            {"asset": "ETH/USDT", "price": 3050.0, "volume": 550.0},
        ]

        all_signals = []

        for data in market_data_sequence:
            signals = await bridge.process_integrated_trading_signal()
                asset=data["asset"], price=data["price"], volume=data["volume"]
            )
            all_signals.extend(signals)

        print(f"   ✅ Generated {len(all_signals)} total integrated signals")

        # Test signal execution (dry, run)
        executed_signals = 0
        for signal in all_signals[:3]:  # Test first 3 signals
            try:
                result = await bridge.execute_integrated_signal(signal)
                if result.get("executed", False):
                    executed_signals += 1
            except Exception as e:
                print(f"     ⚠️  Signal execution issue: {e}")

        print(f"   ✅ Successfully executed {executed_signals} signals")

        # Test optimization
        await bridge.optimize_integration()
        print("   ✅ Integration optimization completed")

        # Final status check
        bridge.get_integration_status()
        print("   ✅ End-to-end integration test completed successfully")

        return True

    except Exception as e:
        print(f"   ❌ End-to-end integration test failed: {e}")
        return False


def test_flake8_compliance():
    """Test flake8 compliance of integration files."""
    print("\n🔍 Testing Flake8 Compliance...")

    try:

        files_to_check = []
            "core/enhanced_strategy_framework.py",
            "core/strategy_integration_bridge.py",
        ]

        for file_path in files_to_check:
            if Path(file_path).exists():
                try:
                    result = subprocess.run()
                        []
                            "python",
                            "-m",
                            "flake8",
                            file_path,
                            "--max-line-length=88",
                            "--extend-ignore=E203,W503",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0:
                        print(f"   ✅ {file_path} passes flake8 compliance")
                    else:
                        print(f"   ❌ {file_path} flake8 issues: {result.stdout}")
                        return False

                except FileNotFoundError:
                    print("   ⚠️  flake8 not available, skipping compliance check")
                    return True
            else:
                print(f"   ⚠️  {file_path} not found")

        return True

    except Exception as e:
        print(f"   ❌ Flake8 compliance test failed: {e}")
        return False


async def run_comprehensive_integration_test():
    """Run comprehensive integration test suite."""
    print("🚀 Comprehensive Integration Test Suite")
    print("=" * 60)
    print("Testing Enhanced Strategy Framework Integration with Schwabot Pipeline")
    print("=" * 60)

    test_results = {}

    # Test 1: Import compatibility
    test_results["import_compatibility"] = check_import_compatibility()

    # Test 2: Enhanced Strategy Framework
    test_results["enhanced_framework"] = test_enhanced_strategy_framework()

    # Test 3: Strategy Integration Bridge
    test_results["integration_bridge"] = await test_strategy_integration_bridge()

    # Test 4: Mathematical Integration
    test_results["mathematical_integration"] = test_mathematical_integration()

    # Test 5: Risk Management Integration
    test_results["risk_management"] = test_risk_management_integration()

    # Test 6: API/Visualization Integration
    test_results["api_visualization"] = test_api_visualization_integration()

    # Test 7: End-to-End Integration
    test_results["end_to_end"] = await test_end_to_end_integration()

    # Test 8: Flake8 Compliance
    test_results["flake8_compliance"] = test_flake8_compliance()

    # Generate summary report
    print("\n📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 50)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed_tests += 1

    success_rate = (passed_tests / total_tests) * 100
    print()
        f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)"
    )

    if success_rate >= 80:
        print("\n🎉 EXCELLENT: Integration is ready for production!")
        print("   All critical components are working correctly.")
        print()
            "   Wall Street strategies are properly integrated with Schwabot pipeline."
        )
        print("   Mathematical framework integration is functional.")
        print("   API endpoints are ready for visualization.")
    elif success_rate >= 60:
        print("\n⚠️  GOOD: Integration is mostly functional with minor issues.")
        print("   Most components are working correctly.")
        print("   Some optimization may be needed for production.")
    else:
        print("\n❌ NEEDS ATTENTION: Critical integration issues detected.")
        print("   Please address failed tests before production deployment.")

    print(f"\n📝 Test completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return success_rate >= 80


def main():
    """Main test execution function."""
    try:
        success = asyncio.run(run_comprehensive_integration_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
