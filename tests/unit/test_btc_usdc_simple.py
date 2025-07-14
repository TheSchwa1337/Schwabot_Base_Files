#!/usr/bin/env python3
"""
Simple BTC/USDC Integration Test
================================

Simplified test for core BTC/USDC trading integration and portfolio balancing.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_system_components():
    """Test basic system component availability."""
    print("🔍 Testing system components...")

    try:
        # Test core imports
        from core.clean_math_foundation import CleanMathFoundation
        from core.clean_profit_vectorization import CleanProfitVectorization
        from core.clean_trading_pipeline import CleanTradingPipeline
        print("✅ Core components imported successfully")

        # Test Phantom Math components
        from core.phantom_detector import PhantomDetector
        from core.phantom_logger import PhantomLogger
        from core.phantom_registry import PhantomRegistry
        print("✅ Phantom Math components imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_portfolio_balancer():
    """Test portfolio balancer functionality."""
    print("\n🔍 Testing portfolio balancer...")

    try:
        from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer

        # Create test config
        config = {}
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.5,
            "max_rebalance_frequency": 60,
        }

        # Create balancer
        balancer = AlgorithmicPortfolioBalancer(config)

        # Check asset allocations
        if not balancer.asset_allocations:
            print("❌ No asset allocations configured")
            return False

        btc_allocation = balancer.asset_allocations.get("BTC")
        if not btc_allocation:
            print("❌ BTC allocation not found")
            return False

        print(f"✅ Portfolio balancer initialized with {len(balancer.asset_allocations)} assets")
        print(f"✅ BTC target weight: {btc_allocation.target_weight}")

        return True

    except Exception as e:
        print(f"❌ Error testing portfolio balancer: {e}")
        return False


def test_btc_usdc_integration():
    """Test BTC/USDC integration functionality."""
    print("\n🔍 Testing BTC/USDC integration...")

    try:
        from core.btc_usdc_trading_integration import BTCUSDCTradingIntegration

        # Create test config
        config = {}
            "btc_usdc_config": {}
                "symbol": "BTC/USDC",
                "base_order_size": 0.01,
                "max_order_size": 0.1,
                "enable_portfolio_balancing": True,
            },
            "portfolio_config": {}
                "rebalancing_strategy": "phantom_adaptive",
                "rebalance_threshold": 0.5,
                "max_rebalance_frequency": 60,
            },
            "exchange_config": {}
                "exchange": "binance",
                "sandbox": True,
            }
        }

        # Create integration
        integration = BTCUSDCTradingIntegration(config)

        # Check configuration
        if integration.config.symbol != "BTC/USDC":
            print("❌ Incorrect symbol configuration")
            return False

        print(f"✅ BTC/USDC integration initialized: {integration.config.symbol}")
        print(f"✅ Base order size: {integration.config.base_order_size}")
        print(f"✅ Max order size: {integration.config.max_order_size}")

        return True

    except Exception as e:
        print(f"❌ Error testing BTC/USDC integration: {e}")
        return False


async def test_phantom_math():
    """Test Phantom Math functionality."""
    print("\n🔍 Testing Phantom Math...")

    try:
        from core.phantom_detector import PhantomDetector
        from core.phantom_logger import PhantomLogger
        from core.phantom_registry import PhantomRegistry

        # Initialize components
        detector = PhantomDetector()
        registry = PhantomRegistry()
        logger = PhantomLogger()

        # Test market data
        market_data = {}
            "BTC": {"price": 50000.0, "volume": 2000000, "timestamp": time.time()},
            "ETH": {"price": 3000.0, "volume": 1500000, "timestamp": time.time()}
        }

        # Detect Phantom Zones
        zones = await detector.detect_phantom_zones(market_data)

        print(f"✅ Phantom Math: {len(zones)} zones detected")

        # Log zones
        for zone in zones:
            await logger.log_phantom_zone(zone)
            await registry.register_phantom_zone(zone)

        print("✅ Phantom Zones logged and registered")

        return True

    except Exception as e:
        print(f"❌ Error testing Phantom Math: {e}")
        return False


async def test_portfolio_state():
    """Test portfolio state management."""
    print("\n🔍 Testing portfolio state management...")

    try:
        from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer

        # Create balancer
        config = {}
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.5,
            "max_rebalance_frequency": 60,
        }
        balancer = AlgorithmicPortfolioBalancer(config)

        # Set initial portfolio state
        balancer.portfolio_state.asset_balances = {}
            "BTC": 0.5,  # 0.5 BTC
            "ETH": 2.0,  # 2.0 ETH
            "USDC": 10000.0  # $10,00 USDC
        }

        # Update with market data
        market_data = {}
            "BTC": {"price": 50000.0, "volume": 2000000},
            "ETH": {"price": 3000.0, "volume": 1500000},
            "USDC": {"price": 1.0, "volume": 5000000}
        }

        await balancer.update_portfolio_state(market_data)

        # Check portfolio state
        total_value = float(balancer.portfolio_state.total_value)
        btc_weight = balancer.portfolio_state.asset_weights.get("BTC", 0)

        print(f"✅ Portfolio total value: ${total_value:,.2f}")
        print(f"✅ BTC weight: {btc_weight:.3f}")

        return total_value > 0 and btc_weight > 0

    except Exception as e:
        print(f"❌ Error testing portfolio state: {e}")
        return False


async def test_rebalancing():
    """Test portfolio rebalancing logic."""
    print("\n🔍 Testing portfolio rebalancing...")

    try:
        from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer

        # Create balancer
        config = {}
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.5,
            "max_rebalance_frequency": 60,
        }
        balancer = AlgorithmicPortfolioBalancer(config)

        # Set unbalanced portfolio
        balancer.portfolio_state.asset_balances = {}
            "BTC": 1.0,  # Overweight BTC
            "ETH": 1.0,  # Underweight ETH
            "USDC": 5000.0  # Underweight USDC
        }

        market_data = {}
            "BTC": {"price": 50000.0, "volume": 2000000},
            "ETH": {"price": 3000.0, "volume": 1500000},
            "USDC": {"price": 1.0, "volume": 5000000}
        }

        await balancer.update_portfolio_state(market_data)

        # Check if rebalancing is needed
        needs_rebalancing = await balancer.check_rebalancing_needs()

        if needs_rebalancing:
            # Generate rebalancing decisions
            decisions = await balancer.generate_rebalancing_decisions(market_data)

            print(f"✅ Rebalancing needed: {needs_rebalancing}")
            print(f"✅ Generated {len(decisions)} rebalancing decisions")

            for decision in decisions:
                print(f"  {decision.symbol}: {decision.action.value} {decision.quantity}")

            return len(decisions) > 0
        else:
            print("✅ No rebalancing needed")
            return True

    except Exception as e:
        print(f"❌ Error testing rebalancing: {e}")
        return False


async def main():
    """Main test runner."""
    print("🚀 Starting Simple BTC/USDC Integration Tests")
    print("=" * 50)

    # Run synchronous tests
    tests = []
        ("System Components", test_system_components),
        ("Portfolio Balancer", test_portfolio_balancer),
        ("BTC/USDC Integration", test_btc_usdc_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    # Run asynchronous tests
    async_tests = []
        ("Phantom Math", test_phantom_math),
        ("Portfolio State", test_portfolio_state),
        ("Rebalancing", test_rebalancing),
    ]

    for test_name, test_func in async_tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

        total += 1

    # Print summary
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error in test suite: {e}")
        sys.exit(1) 