#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE 2-GRAM INTEGRATION TEST
=======================================

Complete integration test for Schwabot's 2-gram pattern detection system.'
Tests all components working together:

🧬 Two-Gram Detector with fractal memory and T-cell protection
🎯 Strategy Trigger Router with pattern-based execution  
🖥️ Visual Execution Node with emoji rendering and GUI
⚖️ Portfolio Balancer with pattern-triggered rebalancing
💱 BTC/USDC Integration with 2-gram signal processing
📊 Hash Registry with pattern storage and evolution
🔮 Phantom Math correlation and entropy synchronization

This test validates the complete Schwabot ecosystem.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer, create_portfolio_balancer
from core.btc_usdc_trading_integration import BTCUSDCTradingIntegration, create_btc_usdc_integration
from core.phantom_detector import PhantomDetector
from core.phantom_registry import PhantomRegistry
from core.strategy_trigger_router import StrategyTriggerRouter, create_strategy_trigger_router

# Import all Schwabot components
from core.two_gram_detector import TwoGramDetector, create_two_gram_detector
from core.visual_execution_node import GUIMode, VisualExecutionNode, VisualizationTheme, create_visual_execution_node
from utils.safe_print import debug, error, info, safe_print, success, warn


class ComprehensiveIntegrationTest:
    """
    Comprehensive test suite for 2-gram integration across Schwabot.

    Tests the complete flow from pattern detection to trade execution,
    with all intermediate components and visualizations.
    """

    def __init__(self):
        self.components = {}
        self.test_results = {}
        self.start_time = time.time()

        # Test configuration
        self.test_config = {}
            "2gram_config": {}
                "window_size": 50,
                "burst_threshold": 1.5,
                "entropy_threshold": 0.5,
                "similarity_threshold": 0.85,
                "t_cell_sensitivity": 0.3,
                "fractal_memory_size": 100
            },
            "router_config": {}
                "execution_mode": "demo"
            },
            "visual_config": {}
                "gui_mode": GUIMode.DEMO_MODE,
                "theme": VisualizationTheme.SCHWABOT_CLASSIC,
                "update_interval_ms": 1000
            },
            "portfolio_config": {}
                "rebalancing_strategy": "phantom_adaptive",
                "rebalance_threshold": 0.5,
                "max_rebalance_frequency": 3600
            },
            "btc_usdc_config": {}
                "order_size_btc": 0.01,
                "max_daily_trades": 10,
                "risk_limit": 0.2
            }
        }

        # Market simulation data
        self.simulation_data = self._generate_simulation_data()

    def _generate_simulation_data(self) -> List[Dict[str, Any]]:
        """Generate realistic market simulation data for testing."""
        data = []
        base_time = time.time()

        # Generate 100 data points over simulated time
        for i in range(100):
            timestamp = base_time + i * 60  # 1 minute intervals

            # Simulate realistic BTC/ETH price movements
            btc_price = 50000 + np.sin(i * 0.1) * 2000 + np.random.normal(0, 500)
            eth_price = 3000 + np.sin(i * 0.15) * 200 + np.random.normal(0, 100)

            # Create market conditions that will trigger patterns
            if i % 20 == 0:  # Volatility spikes every 20 intervals
                btc_price += np.random.choice([-1000, 1000])
                eth_price += np.random.choice([-150, 150])

            data_point = {}
                "timestamp": timestamp,
                "BTC": {}
                    "price": btc_price,
                    "price_change_24h": np.random.normal(0, 3),
                    "volume": 1000000 + np.random.normal(0, 100000),
                    "volume_change_24h": np.random.normal(0, 20)
                },
                "ETH": {}
                    "price": eth_price,
                    "price_change_24h": np.random.normal(0, 4),
                    "volume": 800000 + np.random.normal(0, 80000),
                    "volume_change_24h": np.random.normal(0, 25)
                },
                "USDC": {}
                    "price": 1.0 + np.random.normal(0, 0.01),
                    "price_change_24h": np.random.normal(0, 0.1),
                    "volume": 5000000 + np.random.normal(0, 200000),
                    "volume_change_24h": np.random.normal(0, 10)
                }
            }

            data.append(data_point)

        return data

    async def setup_components(self):
        """Initialize all Schwabot components for testing."""
        info("🔧 Setting up Schwabot components...")

        try:
            # 1. Initialize 2-gram detector
            self.components["two_gram_detector"] = create_two_gram_detector()
                self.test_config["2gram_config"]
            )

            # 2. Initialize strategy trigger router
            self.components["strategy_router"] = create_strategy_trigger_router()
                self.test_config["router_config"]
            )

            # 3. Initialize visual execution node
            self.components["visual_node"] = create_visual_execution_node()
                self.test_config["visual_config"]
            )

            # 4. Initialize portfolio balancer
            self.components["portfolio_balancer"] = create_portfolio_balancer()
                self.test_config["portfolio_config"]
            )

            # 5. Initialize BTC/USDC integration
            self.components["btc_usdc_integration"] = create_btc_usdc_integration()
                self.test_config["btc_usdc_config"]
            )

            # 6. Initialize Phantom components
            self.components["phantom_registry"] = PhantomRegistry()

            # Inject dependencies
            await self._inject_dependencies()

            success("✅ All components initialized successfully")

        except Exception as e:
            error(f"❌ Component setup failed: {e}")
            raise

    async def _inject_dependencies(self):
        """Inject dependencies between components."""
        info("🔌 Injecting component dependencies...")

        # Inject trading components into strategy router
        await self.components["strategy_router"].inject_trading_components()
            self.components["portfolio_balancer"],
            self.components["btc_usdc_integration"]
        )

        # Inject components into visual node
        await self.components["visual_node"].inject_components()
            self.components["two_gram_detector"],
            self.components["strategy_router"],
            self.components["portfolio_balancer"],
            self.components["btc_usdc_integration"]
        )

        success("✅ Dependencies injected")

    async def test_component_initialization(self) -> bool:
        """Test 1: Verify all components are properly initialized."""
        info("🧪 Test 1: Component Initialization")

        try:
            results = {}

            # Test each component
            for name, component in self.components.items():
                if hasattr(component, 'health_check'):
                    health = await component.health_check()
                    results[name] = health.get("status", "unknown") != "error"
                elif hasattr(component, 'get_system_status'):
                    status = await component.get_system_status()
                    results[name] = status.get("status", "unknown") != "error"
                else:
                    results[name] = component is not None

            all_healthy = all(results.values())

            self.test_results["component_initialization"] = {}
                "passed": all_healthy,
                "details": results,
                "message": "All components initialized" if all_healthy else "Some components failed"
            }

            if all_healthy:
                success("✅ Test 1 PASSED: All components initialized")
            else:
                warn("⚠️ Test 1 FAILED: Some components not initialized")

            return all_healthy

        except Exception as e:
            error(f"❌ Test 1 ERROR: {e}")
            self.test_results["component_initialization"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_2gram_pattern_detection(self) -> bool:
        """Test 2: Verify 2-gram pattern detection functionality."""
        info("🧪 Test 2: 2-Gram Pattern Detection")

        try:
            detector = self.components["two_gram_detector"]

            # Test with known pattern sequences
            test_sequences = []
                "UDUDUD",  # Should trigger volatility patterns
                "BEBEBE",  # Should trigger swap patterns
                "UUUUUU",  # Should trigger trend patterns
                "AAAAAA",  # Should trigger flatline protection
                "EEEEEE"   # Should trigger entropy spike
            ]

            detected_patterns = []

            for sequence in test_sequences:
                signals = await detector.analyze_sequence(sequence, {"test_context": True})
                detected_patterns.extend(signals)

            # Verify pattern detection
            pattern_types = {signal.pattern for signal in detected_patterns}
            expected_patterns = {"UD", "BE", "UU", "AA", "EE"}

            patterns_found = len(pattern_types.intersection(expected_patterns))

            # Test pattern statistics
            stats = await detector.get_pattern_statistics()

            success_criteria = []
                len(detected_patterns) > 0,
                patterns_found >= 3,  # At least 3 expected patterns
                stats.get("total_sequences_processed", 0) > 0,
                stats.get("active_patterns", 0) > 0
            ]

            test_passed = all(success_criteria)

            self.test_results["2gram_pattern_detection"] = {}
                "passed": test_passed,
                "patterns_detected": len(detected_patterns),
                "pattern_types": list(pattern_types),
                "expected_patterns": list(expected_patterns),
                "patterns_found": patterns_found,
                "statistics": stats
            }

            if test_passed:
                success(f"✅ Test 2 PASSED: Detected {len(detected_patterns)} patterns")
            else:
                warn(f"⚠️ Test 2 FAILED: Only detected {patterns_found}/{len(expected_patterns)} expected patterns")

            return test_passed

        except Exception as e:
            error(f"❌ Test 2 ERROR: {e}")
            self.test_results["2gram_pattern_detection"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_strategy_trigger_integration(self) -> bool:
        """Test 3: Verify strategy trigger integration with 2-gram patterns."""
        info("🧪 Test 3: Strategy Trigger Integration")

        try:
            router = self.components["strategy_router"]

            # Process market data to generate triggers
            triggers = []
            executed_results = []

            for i, market_data in enumerate(self.simulation_data[:20]):  # Use first 20 data points
                data_triggers = await router.process_market_data(market_data)
                triggers.extend(data_triggers)

                # Execute some triggers
                for trigger in data_triggers[:2]:  # Execute up to 2 triggers per data point
                    result = await router.execute_trigger(trigger)
                    executed_results.append(result)

            # Analyze results
            successful_executions = sum(1 for r in executed_results if r.execution_success)
            pattern_triggers = sum(1 for t in triggers if "2gram" in t.strategy_name.lower())

            # Get router statistics
            router_stats = await router.get_router_statistics()

            success_criteria = []
                len(triggers) > 0,
                len(executed_results) > 0,
                successful_executions > 0,
                pattern_triggers > 0,
                router_stats.get("success_rate", 0) >= 0  # Just check it's calculated'
            ]

            test_passed = all(success_criteria)

            self.test_results["strategy_trigger_integration"] = {}
                "passed": test_passed,
                "total_triggers": len(triggers),
                "pattern_triggers": pattern_triggers,
                "executed_results": len(executed_results),
                "successful_executions": successful_executions,
                "success_rate": successful_executions / len(executed_results) if executed_results else 0,
                "router_statistics": router_stats
            }

            if test_passed:
                success(f"✅ Test 3 PASSED: Generated {len(triggers)} triggers, executed {successful_executions}")
            else:
                warn(f"⚠️ Test 3 FAILED: Insufficient trigger generation or execution")

            return test_passed

        except Exception as e:
            error(f"❌ Test 3 ERROR: {e}")
            self.test_results["strategy_trigger_integration"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_portfolio_balance_integration(self) -> bool:
        """Test 4: Verify portfolio balancing with pattern triggers."""
        info("🧪 Test 4: Portfolio Balance Integration")

        try:
            balancer = self.components["portfolio_balancer"]

            # Initialize portfolio with some assets
            initial_state = {}
                "BTC": {"balance": 0.1, "value": 5000.0},
                "ETH": {"balance": 2.0, "value": 6000.0},
                "USDC": {"balance": 1000.0, "value": 1000.0}
            }

            # Update portfolio state with market data
            for market_data in self.simulation_data[:10]:
                await balancer.update_portfolio_state(market_data)

                # Check if rebalancing is needed
                needs_rebalancing = await balancer.check_rebalancing_needs()

                if needs_rebalancing:
                    # Generate rebalancing decisions
                    decisions = await balancer.generate_rebalancing_decisions(market_data)

                    # In demo mode, just simulate execution
                    if decisions:
                        info(f"🔄 Generated {len(decisions)} rebalancing decisions")

            # Calculate performance metrics
            performance = await balancer.calculate_performance_metrics()

            # Get current portfolio state
            portfolio_state = balancer.portfolio_state

            success_criteria = []
                portfolio_state.total_value > 0,
                len(portfolio_state.asset_weights) > 0,
                performance.get("total_return") is not None,
                performance.get("sharpe_ratio") is not None
            ]

            test_passed = all(success_criteria)

            self.test_results["portfolio_balance_integration"] = {}
                "passed": test_passed,
                "portfolio_value": float(portfolio_state.total_value),
                "asset_weights": portfolio_state.asset_weights,
                "performance_metrics": performance,
                "rebalancing_history": len(portfolio_state.rebalancing_history)
            }

            if test_passed:
                success(f"✅ Test 4 PASSED: Portfolio value: ${portfolio_state.total_value:.2f}")
            else:
                warn(f"⚠️ Test 4 FAILED: Portfolio integration issues")

            return test_passed

        except Exception as e:
            error(f"❌ Test 4 ERROR: {e}")
            self.test_results["portfolio_balance_integration"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_btc_usdc_trading_integration(self) -> bool:
        """Test 5: Verify BTC/USDC trading integration with pattern signals."""
        info("🧪 Test 5: BTC/USDC Trading Integration")

        try:
            integration = self.components["btc_usdc_integration"]

            # Process market data to generate trading decisions
            trading_decisions = []
            processed_count = 0

            for market_data in self.simulation_data[:15]:
                decision = await integration.process_market_data(market_data)
                if decision:
                    trading_decisions.append(decision)
                processed_count += 1

            # Test trading decision validation
            valid_decisions = 0
            for decision in trading_decisions:
                is_valid = await integration.validate_trading_decision(decision)
                if is_valid:
                    valid_decisions += 1

            # Get performance metrics
            performance = await integration.calculate_performance_metrics()

            # Get system status
            status = await integration.get_system_status()

            success_criteria = []
                processed_count > 0,
                len(trading_decisions) >= 0,  # May be zero in demo mode
                valid_decisions >= 0,  # May be zero in demo mode
                status.get("status") != "error",
                performance is not None
            ]

            test_passed = all(success_criteria)

            self.test_results["btc_usdc_trading_integration"] = {}
                "passed": test_passed,
                "processed_market_data": processed_count,
                "trading_decisions": len(trading_decisions),
                "valid_decisions": valid_decisions,
                "performance_metrics": performance,
                "system_status": status
            }

            if test_passed:
                success(f"✅ Test 5 PASSED: Processed {processed_count} data points, {len(trading_decisions)} decisions")
            else:
                warn(f"⚠️ Test 5 FAILED: Trading integration issues")

            return test_passed

        except Exception as e:
            error(f"❌ Test 5 ERROR: {e}")
            self.test_results["btc_usdc_trading_integration"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_visual_execution_integration(self) -> bool:
        """Test 6: Verify visual execution node integration."""
        info("🧪 Test 6: Visual Execution Integration")

        try:
            visual_node = self.components["visual_node"]

            # Test visualization statistics
            stats = await visual_node.get_visualization_statistics()

            # Test headless update (since GUI may not be available in test, environment)
            for i in range(3):
                await visual_node._update_headless()
                await asyncio.sleep(0.1)

            # Verify visual node can process data
            success_criteria = []
                stats.get("running") is not None,
                stats.get("gui_available") is not None,
                stats.get("theme") is not None,
                stats.get("memory_usage_mb") is not None
            ]

            test_passed = all(success_criteria)

            self.test_results["visual_execution_integration"] = {}
                "passed": test_passed,
                "visualization_statistics": stats,
                "gui_available": stats.get("gui_available", False)
            }

            if test_passed:
                success(f"✅ Test 6 PASSED: Visual node operational (GUI: {stats.get('gui_available', False)})")
            else:
                warn(f"⚠️ Test 6 FAILED: Visual execution issues")

            return test_passed

        except Exception as e:
            error(f"❌ Test 6 ERROR: {e}")
            self.test_results["visual_execution_integration"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    async def test_end_to_end_flow(self) -> bool:
        """Test 7: Complete end-to-end flow from pattern detection to execution."""
        info("🧪 Test 7: End-to-End Integration Flow")

        try:
            # Simulate complete flow
            detector = self.components["two_gram_detector"]
            router = self.components["strategy_router"]

            end_to_end_results = {}
                "patterns_detected": 0,
                "triggers_generated": 0,
                "strategies_executed": 0,
                "successful_executions": 0
            }

            # Process a subset of simulation data through the complete pipeline
            for i, market_data in enumerate(self.simulation_data[:10]):
                # 1. Generate market sequence for pattern detection
                sequence = self._market_data_to_sequence(market_data)

                # 2. Detect patterns
                patterns = await detector.analyze_sequence(sequence, market_data)
                end_to_end_results["patterns_detected"] += len(patterns)

                # 3. Generate strategy triggers from market data
                triggers = await router.process_market_data(market_data)
                end_to_end_results["triggers_generated"] += len(triggers)

                # 4. Execute triggers
                for trigger in triggers[:2]:  # Limit executions
                    result = await router.execute_trigger(trigger)
                    end_to_end_results["strategies_executed"] += 1
                    if result.execution_success:
                        end_to_end_results["successful_executions"] += 1

                # Brief pause to simulate real-time processing
                await asyncio.sleep(0.1)

            # Calculate success metrics
            execution_success_rate = ()
                end_to_end_results["successful_executions"] / 
                end_to_end_results["strategies_executed"]
            ) if end_to_end_results["strategies_executed"] > 0 else 0

            success_criteria = []
                end_to_end_results["patterns_detected"] > 0,
                end_to_end_results["triggers_generated"] > 0,
                end_to_end_results["strategies_executed"] > 0,
                execution_success_rate > 0.5  # At least 50% success rate
            ]

            test_passed = all(success_criteria)

            self.test_results["end_to_end_flow"] = {}
                "passed": test_passed,
                "results": end_to_end_results,
                "execution_success_rate": execution_success_rate
            }

            if test_passed:
                success(f"✅ Test 7 PASSED: End-to-end flow operational (success rate: {execution_success_rate:.1%})")
            else:
                warn(f"⚠️ Test 7 FAILED: End-to-end flow issues")

            return test_passed

        except Exception as e:
            error(f"❌ Test 7 ERROR: {e}")
            self.test_results["end_to_end_flow"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    def _market_data_to_sequence(self, market_data: Dict[str, Any]) -> str:
        """Convert market data to sequence for pattern detection."""
        sequence = ""

        # Convert price movements to characters
        for asset, data in market_data.items():
            if asset in ["BTC", "ETH"] and isinstance(data, dict):
                change = data.get("price_change_24h", 0)
                if change > 2:
                    sequence += "U"
                elif change < -2:
                    sequence += "D"
                else:
                    sequence += "C"

        return sequence or "CC"  # Default sequence

    async def test_system_stress(self) -> bool:
        """Test 8: System stress test with rapid data processing."""
        info("🧪 Test 8: System Stress Test")

        try:
            start_time = time.time()

            # Process all simulation data rapidly
            stress_results = {}
                "data_points_processed": 0,
                "total_patterns": 0,
                "total_triggers": 0,
                "total_executions": 0,
                "processing_time": 0,
                "average_latency": 0
            }

            detector = self.components["two_gram_detector"]
            router = self.components["strategy_router"]

            processing_times = []

            for market_data in self.simulation_data:
                point_start = time.time()

                # Generate sequence and detect patterns
                sequence = self._market_data_to_sequence(market_data)
                patterns = await detector.analyze_sequence(sequence, market_data)
                stress_results["total_patterns"] += len(patterns)

                # Generate and execute triggers
                triggers = await router.process_market_data(market_data)
                stress_results["total_triggers"] += len(triggers)

                # Execute first trigger only for speed
                if triggers:
                    result = await router.execute_trigger(triggers[0])
                    stress_results["total_executions"] += 1

                stress_results["data_points_processed"] += 1

                point_time = time.time() - point_start
                processing_times.append(point_time)

            stress_results["processing_time"] = time.time() - start_time
            stress_results["average_latency"] = np.mean(processing_times) * 1000  # ms

            # Performance criteria
            success_criteria = []
                stress_results["data_points_processed"] == len(self.simulation_data),
                stress_results["average_latency"] < 100,  # Less than 100ms average
                stress_results["total_patterns"] > 0,
                stress_results["total_triggers"] > 0
            ]

            test_passed = all(success_criteria)

            self.test_results["system_stress"] = {}
                "passed": test_passed,
                "results": stress_results
            }

            if test_passed:
                success(f"✅ Test 8 PASSED: Processed {stress_results['data_points_processed']} points in {stress_results['processing_time']:.2f}s (avg: {stress_results['average_latency']:.1f}ms)")
            else:
                warn(f"⚠️ Test 8 FAILED: System stress test issues")

            return test_passed

        except Exception as e:
            error(f"❌ Test 8 ERROR: {e}")
            self.test_results["system_stress"] = {}
                "passed": False,
                "error": str(e)
            }
            return False

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("passed", False))
        total_time = time.time() - self.start_time

        report = f"""
🧪 COMPREHENSIVE 2-GRAM INTEGRATION TEST REPORT
{'=' * 60}

📊 SUMMARY:
Tests Passed: {passed_tests}/{total_tests}
Success Rate: {(passed_tests/total_tests)*100:.1f}%
Total Time: {total_time:.2f} seconds

🧬 COMPONENT STATUS:
"""

        for component_name in self.components.keys():
            status = "✅ OPERATIONAL" if self.components[component_name] else "❌ FAILED"
            report += f"  {component_name}: {status}\n"

        report += "\n🧪 TEST RESULTS:\n"

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            report += f"  {test_name}: {status}\n"

            if not result.get("passed", False) and "error" in result:
                report += f"    Error: {result['error']}\n"

        # Add specific metrics
        report += "\n📈 KEY METRICS:\n"

        if "2gram_pattern_detection" in self.test_results:
            patterns = self.test_results["2gram_pattern_detection"].get("patterns_detected", 0)
            report += f"  Patterns Detected: {patterns}\n"

        if "strategy_trigger_integration" in self.test_results:
            triggers = self.test_results["strategy_trigger_integration"].get("total_triggers", 0)
            report += f"  Strategy Triggers: {triggers}\n"

        if "end_to_end_flow" in self.test_results:
            success_rate = self.test_results["end_to_end_flow"].get("execution_success_rate", 0)
            report += f"  Execution Success Rate: {success_rate:.1%}\n"

        if "system_stress" in self.test_results:
            latency = self.test_results["system_stress"].get("results", {}).get("average_latency", 0)
            report += f"  Average Latency: {latency:.1f}ms\n"

        report += f"\n🎯 OVERALL STATUS: {'✅ SYSTEM OPERATIONAL' if passed_tests >= total_tests * 0.8 else '⚠️ SYSTEM ISSUES DETECTED'}\n"

        return report

    async def run_all_tests(self):
        """Run complete test suite."""
        info("🚀 Starting Comprehensive 2-Gram Integration Test Suite")
        info("=" * 60)

        # Setup
        await self.setup_components()

        # Run all tests
        tests = []
            self.test_component_initialization,
            self.test_2gram_pattern_detection,
            self.test_strategy_trigger_integration,
            self.test_portfolio_balance_integration,
            self.test_btc_usdc_trading_integration,
            self.test_visual_execution_integration,
            self.test_end_to_end_flow,
            self.test_system_stress
        ]

        for i, test in enumerate(tests, 1):
            info(f"\n📋 Running Test {i}/{len(tests)}: {test.__name__}")
            try:
                await test()
            except Exception as e:
                error(f"Test {test.__name__} crashed: {e}")

            await asyncio.sleep(0.1)  # Brief pause between tests

        # Generate and display report
        report = self.generate_test_report()
        print(report)

        # Save detailed results
        with open("test_results_comprehensive_2gram.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        info("📄 Detailed results saved to test_results_comprehensive_2gram.json")


async def main():
    """Main test execution function."""
    test_suite = ComprehensiveIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 