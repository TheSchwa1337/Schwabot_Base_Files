#!/usr/bin/env python3
"""
Test BTC/USDC Integration and Portfolio Balancing
=================================================

Comprehensive test script to validate:
- Algorithmic portfolio balancer
- BTC/USDC trading integration
- Phantom Math integration
- System initialization and operation
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

    create_clean_trading_system,
    get_system_status,
    PORTFOLIO_BALANCER_AVAILABLE,
    BTC_USDC_INTEGRATION_AVAILABLE,
)
from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer, create_portfolio_balancer
from core.btc_usdc_trading_integration import BTCUSDCTradingIntegration, create_btc_usdc_integration
from core.phantom_detector import PhantomDetector
from core.phantom_logger import PhantomLogger
from core.phantom_registry import PhantomRegistry
from utils.safe_print import error, info, safe_print, success, warn

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Test suite for BTC/USDC integration and portfolio balancing."""

    def __init__(self):
        self.test_results = {}
        self.config = self._create_test_config()

    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {}
            "portfolio_config": {}
                "rebalancing_strategy": "phantom_adaptive",
                "rebalance_threshold": 0.5,
                "max_rebalance_frequency": 60,  # 1 minute for testing
            },
            "btc_usdc_config": {}
                "symbol": "BTC/USDC",
                "base_order_size": 0.01,
                "max_order_size": 0.1,
                "enable_portfolio_balancing": True,
                "max_daily_trades": 10,  # Lower for testing
            },
            "exchange_config": {}
                "exchange": "binance",
                "sandbox": True,
            }
        }

    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        info("🧪 Starting BTC/USDC Integration and Portfolio Balancing Tests")

        tests = []
            ("System Status Check", self.test_system_status),
            ("Portfolio Balancer Initialization", self.test_portfolio_balancer_init),
            ("BTC/USDC Integration Initialization", self.test_btc_usdc_init),
            ("Phantom Math Integration", self.test_phantom_math_integration),
            ("Portfolio State Management", self.test_portfolio_state_management),
            ("Rebalancing Logic", self.test_rebalancing_logic),
            ("Trading Decision Generation", self.test_trading_decision_generation),
            ("Market Data Processing", self.test_market_data_processing),
            ("Performance Metrics", self.test_performance_metrics),
            ("Full System Integration", self.test_full_system_integration),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            info(f"\n🔍 Running: {test_name}")
            try:
                result = await test_func()
                if result:
                    success(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    error(f"❌ {test_name}: FAILED")
            except Exception as e:
                error(f"❌ {test_name}: ERROR - {e}")

            self.test_results[test_name] = result

        # Print summary
        info(f"\n📊 Test Summary: {passed}/{total} tests passed")
        if passed == total:
            success("🎉 All tests passed! Integration is working correctly.")
        else:
            warn(f"⚠️ {total - passed} tests failed. Check the logs for details.")

        return passed == total

    async def test_system_status(self) -> bool:
        """Test system component availability."""
        try:
            status = get_system_status()

            # Check required components
            required_components = []
                "clean_math_foundation",
                "clean_profit_vectorization", 
                "clean_trading_pipeline",
                "portfolio_balancer",
                "btc_usdc_integration"
            ]

            for component in required_components:
                if not status["clean_implementations"].get(component, False):
                    error(f"Missing required component: {component}")
                    return False

            info(f"System status: {status['system_operational']}")
            return status["system_operational"]

        except Exception as e:
            error(f"Error checking system status: {e}")
            return False

    async def test_portfolio_balancer_init(self) -> bool:
        """Test portfolio balancer initialization."""
        try:
            if not PORTFOLIO_BALANCER_AVAILABLE:
                error("Portfolio balancer not available")
                return False

            balancer = create_portfolio_balancer(self.config)

            # Check initialization
            if not balancer:
                error("Failed to create portfolio balancer")
                return False

            # Check asset allocations
            if not balancer.asset_allocations:
                error("No asset allocations configured")
                return False

            # Check BTC allocation
            btc_allocation = balancer.asset_allocations.get("BTC")
            if not btc_allocation:
                error("BTC allocation not found")
                return False

            info(f"Portfolio balancer initialized with {len(balancer.asset_allocations)} assets")
            info(f"BTC target weight: {btc_allocation.target_weight}")

            return True

        except Exception as e:
            error(f"Error testing portfolio balancer: {e}")
            return False

    async def test_btc_usdc_init(self) -> bool:
        """Test BTC/USDC integration initialization."""
        try:
            if not BTC_USDC_INTEGRATION_AVAILABLE:
                error("BTC/USDC integration not available")
                return False

            integration = create_btc_usdc_integration(self.config)

            # Check initialization
            if not integration:
                error("Failed to create BTC/USDC integration")
                return False

            # Check configuration
            if integration.config.symbol != "BTC/USDC":
                error("Incorrect symbol configuration")
                return False

            info(f"BTC/USDC integration initialized: {integration.config.symbol}")
            info(f"Base order size: {integration.config.base_order_size}")
            info(f"Max order size: {integration.config.max_order_size}")

            return True

        except Exception as e:
            error(f"Error testing BTC/USDC integration: {e}")
            return False

    async def test_phantom_math_integration(self) -> bool:
        """Test Phantom Math integration."""
        try:
            # Initialize Phantom Math components
            detector = PhantomDetector()
            registry = PhantomRegistry()
            logger = PhantomLogger()

            # Test Phantom Zone detection
            market_data = {}
                "BTC": {"price": 50000.0, "volume": 2000000, "timestamp": time.time()},
                "ETH": {"price": 3000.0, "volume": 1500000, "timestamp": time.time()}
            }

            zones = await detector.detect_phantom_zones(market_data)

            # Log zones
            for zone in zones:
                await logger.log_phantom_zone(zone)
                await registry.register_phantom_zone(zone)

            info(f"Phantom Math integration: {len(zones)} zones detected")

            return True

        except Exception as e:
            error(f"Error testing Phantom Math integration: {e}")
            return False

    async def test_portfolio_state_management(self) -> bool:
        """Test portfolio state management."""
        try:
            balancer = create_portfolio_balancer(self.config)

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

            info(f"Portfolio total value: ${total_value:,.2f}")
            info(f"BTC weight: {btc_weight:.3f}")

            return total_value > 0 and btc_weight > 0

        except Exception as e:
            error(f"Error testing portfolio state management: {e}")
            return False

    async def test_rebalancing_logic(self) -> bool:
        """Test portfolio rebalancing logic."""
        try:
            balancer = create_portfolio_balancer(self.config)

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

                info(f"Rebalancing needed: {needs_rebalancing}")
                info(f"Generated {len(decisions)} rebalancing decisions")

                for decision in decisions:
                    info(f"  {decision.symbol}: {decision.action.value} {decision.quantity}")

                return len(decisions) > 0
            else:
                info("No rebalancing needed")
                return True

        except Exception as e:
            error(f"Error testing rebalancing logic: {e}")
            return False

    async def test_trading_decision_generation(self) -> bool:
        """Test trading decision generation."""
        try:
            integration = create_btc_usdc_integration(self.config)

            # Set up market data
            market_data = {}
                "BTC": {}
                    "price": 50000.0,
                    "volume": 2000000,
                    "timestamp": time.time()
                }
            }

            # Process market data
            decision = await integration.process_market_data(market_data)

            if decision:
                info(f"Trading decision generated: {decision.symbol} {decision.action.value} {decision.quantity}")
                info(f"Confidence: {decision.confidence}")
                info(f"Strategy branch: {decision.strategy_branch}")
                return True
            else:
                info("No trading decision generated (expected in some, cases)")
                return True

        except Exception as e:
            error(f"Error testing trading decision generation: {e}")
            return False

    async def test_market_data_processing(self) -> bool:
        """Test market data processing."""
        try:
            integration = create_btc_usdc_integration(self.config)

            # Test market analysis
            market_data = {}
                "BTC": {}
                    "price": 50000.0,
                    "volume": 2000000,
                    "timestamp": time.time()
                }
            }

            analysis = await integration._analyze_market_conditions()

            info(f"Market analysis: {analysis}")

            # Test Phantom signal checking
            phantom_signal = await integration._check_phantom_signals()
            if phantom_signal:
                info(f"Phantom signal: {phantom_signal}")

            # Test portfolio balancing check
            portfolio_signal = await integration._check_portfolio_balancing()
            if portfolio_signal:
                info(f"Portfolio signal: {portfolio_signal}")

            return True

        except Exception as e:
            error(f"Error testing market data processing: {e}")
            return False

    async def test_performance_metrics(self) -> bool:
        """Test performance metrics calculation."""
        try:
            balancer = create_portfolio_balancer(self.config)
            integration = create_btc_usdc_integration(self.config)

            # Get portfolio metrics
            portfolio_metrics = await balancer.get_portfolio_metrics()
            info(f"Portfolio metrics: {portfolio_metrics}")

            # Get trading metrics
            trading_metrics = await integration.get_performance_metrics()
            info(f"Trading metrics: {trading_metrics}")

            return True

        except Exception as e:
            error(f"Error testing performance metrics: {e}")
            return False

    async def test_full_system_integration(self) -> bool:
        """Test full system integration."""
        try:
            # Create complete trading system
            system = create_clean_trading_system(initial_capital=100000.0)

            # Check all components
            required_components = []
                "math_foundation",
                "profit_vectorizer", 
                "trading_pipeline",
                "portfolio_balancer",
                "btc_usdc_integration"
            ]

            for component in required_components:
                if component not in system:
                    error(f"Missing component in system: {component}")
                    return False

            info("Full system integration successful")
            info(f"System components: {list(system.keys())}")

            return True

        except Exception as e:
            error(f"Error testing full system integration: {e}")
            return False


async def main():
    """Main test runner."""
    info("🚀 Starting BTC/USDC Integration and Portfolio Balancing Tests")

    tester = IntegrationTester()
    success = await tester.run_all_tests()

    if success:
        success("🎉 All integration tests passed!")
        return 0
    else:
        error("❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        error(f"Fatal error in test suite: {e}")
        sys.exit(1) 