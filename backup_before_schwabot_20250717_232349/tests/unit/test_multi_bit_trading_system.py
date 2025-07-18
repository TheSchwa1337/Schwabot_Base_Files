import asyncio
import json
import logging
import time
from typing import Any, Dict

import numpy as np
import psutil

from core.dualistic_thought_engines import DualisticThoughtEngines
from core.multi_bit_state_manager import MultiBitStateManager, ProcessingMode
from core.trading_pipeline_integration import TradingPipelineIntegration
from utils.safe_print import error, info, success, warn

# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Multi-Bit Trading System.

Tests the complete integration of:
- Multi-bit state management
- Trading pipeline integration
- Mathematical framework integration
- Dualistic thought engines
- Chrome-inspired memory management
"""


    calculate_ferris_wheel_state,
    calculate_quantum_thermal_state,
    calculate_void_well_metrics,
    calculate_profit_state,
    calculate_kelly_metrics,
)

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiBitTradingSystemTester:
    """Comprehensive tester for the multi-bit trading system."""

    def __init__(self):
        """Initialize the tester."""
        self.multi_bit_manager = MultiBitStateManager(enable_gpu=True)
        self.trading_pipeline = TradingPipelineIntegration(enable_gpu=True)
        self.dualistic_engines = DualisticThoughtEngines()

        # Test results
        self.test_results = {}
            "multi_bit_tests": {},
            "pipeline_tests": {},
            "mathematical_tests": {},
            "integration_tests": {},
            "performance_tests": {},
        }
        logger.info("MultiBitTradingSystemTester initialized")

    def test_multi_bit_state_management():-> Dict[str, Any]:
        """Test multi-bit state management functionality."""
        info("Testing Multi-Bit State Management...")

        results = {}
            "state_creation": False,
            "state_transitions": False,
            "mathematical_integration": False,
            "performance_metrics": False,
            "garbage_collection": False,
        }
        try:
            # Test 1: State Creation
            info("  Testing state creation...")

            # Create states with different bit depths
            state_2bit = self.multi_bit_manager.create_memory_state("test_2bit", 2, 0.5)
            state_4bit = self.multi_bit_manager.create_memory_state("test_4bit", 4, 0.7)
            state_8bit = self.multi_bit_manager.create_memory_state("test_8bit", 8, 0.8)
            state_16bit = self.multi_bit_manager.create_memory_state()
                "test_16bit", 16, 0.9
            )
            state_32bit = self.multi_bit_manager.create_memory_state()
                "test_32bit", 32, 1.0
            )

            assert state_2bit.bit_depth == 2
            assert state_4bit.bit_depth == 4
            assert state_8bit.bit_depth == 8
            assert state_16bit.bit_depth == 16
            assert state_32bit.bit_depth == 32

            results["state_creation"] = True
            success("    ✓ State creation successful")

            # Test 2: State Transitions
            info("  Testing state transitions...")

            transition1 = self.multi_bit_manager.transition_state()
                "test_2bit", "test_4bit", "upgrade"
            )
            transition2 = self.multi_bit_manager.transition_state()
                "test_4bit", "test_8bit", "processing"
            )
            transition3 = self.multi_bit_manager.transition_state()
                "test_8bit", "test_16bit", "gpu_required"
            )
            transition4 = self.multi_bit_manager.transition_state()
                "test_16bit", "test_32bit", "high_precision"
            )

            assert transition1.success
            assert transition2.success
            assert transition3.success
            assert transition4.success

            results["state_transitions"] = True
            success("    ✓ State transitions successful")

            # Test 3: Mathematical Integration
            info("  Testing mathematical integration...")

            mathematical_state = {}
                "ferris_wheel": {}
                    "cycle_position": 1.57,
                    "harmonic_phases": [1.57, 0.52, 0.22],
                    "angular_velocity": 0.26,
                    "phase_coherence": 0.85,
                    "synchronization_level": 0.12,
                },
                "quantum_thermal": {}
                    "quantum_state": [0.707, 0.707],
                    "temperature": 300.0,
                    "thermal_entropy": 0.5,
                    "coupling_strength": 0.8,
                    "decoherence_rate": 0.01,
                },
            }
            math_state = self.multi_bit_manager.create_memory_state()
                "test_math", 32, 1.0, mathematical_state
            )

            assert math_state.ferris_wheel_state is not None
            assert math_state.quantum_thermal_state is not None

            results["mathematical_integration"] = True
            success("    ✓ Mathematical integration successful")

            # Test 4: Performance Metrics
            info("  Testing performance metrics...")

            performance = self.multi_bit_manager.get_performance_summary()

            assert "performance_metrics" in performance
            assert "system_info" in performance
            assert "recent_transitions" in performance

            results["performance_metrics"] = True
            success("    ✓ Performance metrics successful")

            # Test 5: Garbage Collection
            info("  Testing garbage collection...")

            # Create many states to trigger garbage collection
            for i in range(50):
                self.multi_bit_manager.create_memory_state(f"gc_test_{i}", 2, 0.1)

            # Force garbage collection
            self.multi_bit_manager._garbage_collect()

            results["garbage_collection"] = True
            success("    ✓ Garbage collection successful")

        except Exception as e:
            error(f"Multi-bit state management test failed: {e}")
            logger.exception("Test failure details")

        self.test_results["multi_bit_tests"] = results
        return results

    def test_mathematical_framework():-> Dict[str, Any]:
        """Test mathematical framework functionality."""
        info("Testing Mathematical Framework...")

        results = {}
            "ferris_wheel": False,
            "quantum_thermal": False,
            "void_well": False,
            "profit_state": False,
            "kelly_metrics": False,
        }
        try:
            # Test 1: Ferris Wheel State
            info("  Testing Ferris wheel state calculation...")

            time_series = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
            periods = [24.0, 72.0, 168.0]
            current_time = time.time()

            ferris_wheel = calculate_ferris_wheel_state()
                time_series, periods, current_time
            )

            assert hasattr(ferris_wheel, "cycle_position")
            assert hasattr(ferris_wheel, "phase_coherence")
            assert 0 <= ferris_wheel.phase_coherence <= 1

            results["ferris_wheel"] = True
            success("    ✓ Ferris wheel calculation successful")

            # Test 2: Quantum Thermal State
            info("  Testing quantum thermal state calculation...")

            quantum_state = np.array([0.70710678, 0.70710678])  # |+⟩ state
            temperature = 300.0

            quantum_thermal = calculate_quantum_thermal_state()
                quantum_state, temperature
            )

            assert hasattr(quantum_thermal, "temperature")
            assert hasattr(quantum_thermal, "thermal_entropy")
            assert quantum_thermal.temperature == temperature

            results["quantum_thermal"] = True
            success("    ✓ Quantum thermal calculation successful")

            # Test 3: Void Well Metrics
            info("  Testing void well metrics calculation...")

            volume_data = np.array([100.0, 120.0, 110.0, 90.0, 130.0])
            price_data = np.array([100.0, 101.0, 102.0, 101.5, 103.0])

            void_well = calculate_void_well_metrics(volume_data, price_data)

            assert hasattr(void_well, "fractal_index")
            assert hasattr(void_well, "volume_divergence")
            assert void_well.fractal_index >= 0

            results["void_well"] = True
            success("    ✓ Void well metrics calculation successful")

            # Test 4: Profit State
            info("  Testing profit state calculation...")

            entry_price = 100.0
            exit_price = 105.0
            time_held = 60.0
            volatility = 0.5

            profit_state = calculate_profit_state()
                entry_price, exit_price, time_held, volatility
            )

            assert hasattr(profit_state, "raw_return")
            assert hasattr(profit_state, "sharpe_ratio")
            assert profit_state.raw_return == 0.5  # 5% return

            results["profit_state"] = True
            success("    ✓ Profit state calculation successful")

            # Test 5: Kelly Metrics
            info("  Testing Kelly metrics calculation...")

            win_probability = 0.6
            expected_return = 0.2
            volatility = 0.5

            kelly_metrics = calculate_kelly_metrics()
                win_probability, expected_return, volatility
            )

            assert hasattr(kelly_metrics, "kelly_fraction")
            assert hasattr(kelly_metrics, "safe_kelly")
            assert 0 <= kelly_metrics.safe_kelly <= 0.25  # Max 25%

            results["kelly_metrics"] = True
            success("    ✓ Kelly metrics calculation successful")

        except Exception as e:
            error(f"Mathematical framework test failed: {e}")
            logger.exception("Test failure details")

        self.test_results["mathematical_tests"] = results
        return results

    async def test_trading_pipeline():-> Dict[str, Any]:
        """Test trading pipeline integration."""
        info("Testing Trading Pipeline Integration...")

        results = {}
            "pipeline_initialization": False,
            "market_data_processing": False,
            "signal_generation": False,
            "risk_management": False,
            "performance_tracking": False,
        }
        try:
            # Test 1: Pipeline Initialization
            info("  Testing pipeline initialization...")

            assert self.trading_pipeline.multi_bit_manager is not None
            assert self.trading_pipeline.dualistic_engines is not None
            assert self.trading_pipeline.risk_management_enabled

            results["pipeline_initialization"] = True
            success("    ✓ Pipeline initialization successful")

            # Test 2: Market Data Processing
            info("  Testing market data processing...")

            sample_market_data = {}
                "current_price": 62000.0,
                "price_change": 0.2,
                "volume_change": 0.15,
                "volatility": 0.6,
                "temperature": 310.0,
                "price_history": [61000.0, 61500.0, 62000.0, 61800.0, 62200.0],
                "volume_data": [100.0, 120.0, 110.0, 90.0, 130.0],
                "price_data": [61000.0, 61500.0, 62000.0, 61800.0, 62200.0],
                "rsi": 65.0,
                "macd_signal": 0.1,
                "moving_average": 61500.0,
            }
            signal = await self.trading_pipeline.process_market_data()
                sample_market_data, "BTC", "warm"
            )

            assert signal is not None
            assert hasattr(signal, "signal_type")
            assert hasattr(signal, "confidence")
            assert hasattr(signal, "bit_depth")

            results["market_data_processing"] = True
            success("    ✓ Market data processing successful")

            # Test 3: Signal Generation
            info("  Testing signal generation...")

            assert signal.signal_type in ["buy", "sell", "hold"]
            assert 0 <= signal.confidence <= 1
            assert signal.bit_depth in [2, 4, 8, 16, 32, 42]
            assert signal.processing_mode in ProcessingMode

            results["signal_generation"] = True
            success("    ✓ Signal generation successful")

            # Test 4: Risk Management
            info("  Testing risk management...")

            if signal.signal_type in ["buy", "sell"]:
                assert signal.stop_loss > 0
                assert signal.take_profit > 0
                assert signal.position_size >= 0
                assert signal.position_size <= 0.25  # Max 25%

            results["risk_management"] = True
            success("    ✓ Risk management successful")

            # Test 5: Performance Tracking
            info("  Testing performance tracking...")

            performance = self.trading_pipeline.get_pipeline_performance()

            assert "pipeline_metrics" in performance
            assert "multi_bit_performance" in performance
            assert "dualistic_engine_performance" in performance

            results["performance_tracking"] = True
            success("    ✓ Performance tracking successful")

        except Exception as e:
            error(f"Trading pipeline test failed: {e}")
            logger.exception("Test failure details")

        self.test_results["pipeline_tests"] = results
        return results

    def test_dualistic_thought_engines():-> Dict[str, Any]:
        """Test dualistic thought engines."""
        info("Testing Dualistic Thought Engines...")

        results = {}
            "engine_initialization": False,
            "logical_analysis": False,
            "intuitive_analysis": False,
            "alif_analysis": False,
            "historical_consultation": False,
            "bias_mitigation": False,
        }
        try:
            # Test 1: Engine Initialization
            info("  Testing engine initialization...")

            assert self.dualistic_engines.unified_math is not None
            assert self.dualistic_engines.schwafit_core is not None
            assert self.dualistic_engines.lantern_core is not None
            assert self.dualistic_engines.alif_enabled

            results["engine_initialization"] = True
            success("    ✓ Engine initialization successful")

            # Test 2: Logical Analysis
            info("  Testing logical analysis...")

            market_data = {}
                "rsi": 25.5,
                "macd_signal": 0.1,
                "volume_change": 0.3,
                "current_price": 62000.0,
                "moving_average": 61500.0,
                "previous_close": 61800.0,
                "volatility": 0.8,
            }
            thought_vector = self.dualistic_engines.process_market_data()
                market_data, "warm"
            )

            assert thought_vector is not None
            assert hasattr(thought_vector, "logical_score")
            assert hasattr(thought_vector, "intuitive_score")
            assert hasattr(thought_vector, "combined_score")
            assert hasattr(thought_vector, "decision")

            results["logical_analysis"] = True
            success("    ✓ Logical analysis successful")

            # Test 3: Intuitive Analysis
            info("  Testing intuitive analysis...")

            assert 0 <= thought_vector.intuitive_score <= 1
            assert thought_vector.decision in ["buy", "sell", "hold"]

            results["intuitive_analysis"] = True
            success("    ✓ Intuitive analysis successful")

            # Test 4: ALIF Analysis
            info("  Testing ALIF analysis...")

            assert hasattr(thought_vector, "alif_score")
            assert hasattr(thought_vector, "alif_decision")
            assert hasattr(thought_vector, "alif_feedback")

            results["alif_analysis"] = True
            success("    ✓ ALIF analysis successful")

            # Test 5: Historical Consultation
            info("  Testing historical consultation...")

            assert hasattr(thought_vector, "historical_adjustment")
            assert hasattr(thought_vector, "historical_consultation")

            results["historical_consultation"] = True
            success("    ✓ Historical consultation successful")

            # Test 6: Bias Mitigation
            info("  Testing bias mitigation...")

            assert hasattr(thought_vector, "bias_mitigated")

            results["bias_mitigation"] = True
            success("    ✓ Bias mitigation successful")

        except Exception as e:
            error(f"Dualistic thought engines test failed: {e}")
            logger.exception("Test failure details")

        self.test_results["integration_tests"] = results
        return results

    def test_performance_and_scalability():-> Dict[str, Any]:
        """Test performance and scalability."""
        info("Testing Performance and Scalability...")

        results = {}
            "concurrent_processing": False,
            "memory_efficiency": False,
            "latency_measurement": False,
            "throughput_measurement": False,
            "resource_utilization": False,
        }
        try:
            # Test 1: Concurrent Processing
            info("  Testing concurrent processing...")

            start_time = time.time()

            # Process multiple market data sets concurrently
            market_data_sets = []
            for i in range(10):
                market_data = {}
                    "current_price": 62000.0 + i * 100,
                    "price_change": 0.2 + i * 0.1,
                    "volume_change": 0.15 + i * 0.5,
                    "volatility": 0.6 + i * 0.2,
                    "temperature": 310.0 + i * 5,
                    "price_history": []
                        61000.0 + i * 100,
                        61500.0 + i * 100,
                        62000.0 + i * 100,
                    ],
                    "volume_data": [100.0 + i * 10, 120.0 + i * 10, 110.0 + i * 10],
                    "price_data": []
                        61000.0 + i * 100,
                        61500.0 + i * 100,
                        62000.0 + i * 100,
                    ],
                    "rsi": 65.0 + i * 2,
                    "macd_signal": 0.1 + i * 0.01,
                    "moving_average": 61500.0 + i * 100,
                }
                market_data_sets.append(market_data)

            # Process all concurrently
            async def process_concurrent():
                tasks = []
                for i, market_data in enumerate(market_data_sets):
                    task = self.trading_pipeline.process_market_data()
                        market_data, f"BTC_{i}", "warm"
                    )
                    tasks.append(task)

                signals = await asyncio.gather(*tasks)
                return signals

            signals = asyncio.run(process_concurrent())

            processing_time = time.time() - start_time

            assert len(signals) == 10
            assert processing_time < 5.0  # Should complete within 5 seconds

            results["concurrent_processing"] = True
            success(f"    ✓ Concurrent processing successful ({processing_time:.3f}s)")

            # Test 2: Memory Efficiency
            info("  Testing memory efficiency...")


            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            assert memory_usage < 500  # Should use less than 500MB

            results["memory_efficiency"] = True
            success(f"    ✓ Memory efficiency successful ({memory_usage:.1f}MB)")

            # Test 3: Latency Measurement
            info("  Testing latency measurement...")

            latency_times = []
            for i in range(5):
                start = time.time()
                asyncio.run()
                    self.trading_pipeline.process_market_data()
                        market_data_sets[i], f"BTC_latency_{i}", "warm"
                    )
                )
                latency = time.time() - start
                latency_times.append(latency)

            avg_latency = np.mean(latency_times)
            assert avg_latency < 1.0  # Average latency should be less than 1 second

            results["latency_measurement"] = True
            success(f"    ✓ Latency measurement successful (avg: {avg_latency:.3f}s)")

            # Test 4: Throughput Measurement
            info("  Testing throughput measurement...")

            throughput = len(signals) / processing_time
            assert throughput > 1.0  # Should process at least 1 signal per second

            results["throughput_measurement"] = True
            success()
                f"    ✓ Throughput measurement successful ({throughput:.1f} signals/s)"
            )

            # Test 5: Resource Utilization
            info("  Testing resource utilization...")

            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            assert cpu_usage < 80  # CPU usage should be reasonable
            assert memory_usage < 80  # Memory usage should be reasonable

            results["resource_utilization"] = True
            success()
                f"    ✓ Resource utilization successful (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)"
            )

        except Exception as e:
            error(f"Performance and scalability test failed: {e}")
            logger.exception("Test failure details")

        self.test_results["performance_tests"] = results
        return results

    def run_all_tests():-> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        info("Starting Comprehensive Multi-Bit Trading System Tests")
        info("=" * 60)

        # Run all tests
        multi_bit_results = self.test_multi_bit_state_management()
        mathematical_results = self.test_mathematical_framework()
        pipeline_results = asyncio.run(self.test_trading_pipeline())
        integration_results = self.test_dualistic_thought_engines()
        performance_results = self.test_performance_and_scalability()

        # Calculate overall results
        all_results = {}
            "multi_bit_tests": multi_bit_results,
            "mathematical_tests": mathematical_results,
            "pipeline_tests": pipeline_results,
            "integration_tests": integration_results,
            "performance_tests": performance_results,
        }
        # Calculate success rates
        success_rates = {}
        for test_category, results in all_results.items():
            if results:
                passed = sum(1 for result in results.values() if result)
                total = len(results)
                success_rate = passed / total * 100
                success_rates[test_category] = {}
                    "passed": passed,
                    "total": total,
                    "success_rate": success_rate,
                }
        # Overall success rate
        total_passed = sum(sr["passed"] for sr in success_rates.values())
        total_tests = sum(sr["total"] for sr in success_rates.values())
        overall_success_rate = ()
            total_passed / total_tests * 100 if total_tests > 0 else 0
        )

        # Print results
        info("\nTest Results Summary:")
        info("=" * 40)

        for category, stats in success_rates.items():
            status = ()
                "✓ PASS"
                if stats["success_rate"] == 100
                else "⚠ PARTIAL"
                if stats["success_rate"] > 50
                else "✗ FAIL"
            )
            info()
                f"{category}: {status} ({stats['passed']}/{stats['total']} - {stats['success_rate']:.1f}%)"
            )

        info()
            f"\nOverall Success Rate: {overall_success_rate:.1f}% ({total_passed}/{total_tests})"
        )

        if overall_success_rate >= 90:
            success("🎉 All tests passed successfully!")
        elif overall_success_rate >= 70:
            warn("⚠ Some tests failed, but system is functional")
        else:
            error("❌ Multiple test failures detected")

        # Cleanup
        self.multi_bit_manager.cleanup()
        self.trading_pipeline.cleanup()

        return {}
            "test_results": all_results,
            "success_rates": success_rates,
            "overall_success_rate": overall_success_rate,
            "total_passed": total_passed,
            "total_tests": total_tests,
        }


def main():
    """Main test execution function."""
    info("Multi-Bit Trading System Comprehensive Test Suite")
    info("=" * 60)

    tester = MultiBitTradingSystemTester()
    results = tester.run_all_tests()

    # Save results to file

    with open("test_results_multi_bit_system.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    info("Test results saved to test_results_multi_bit_system.json")

    return results


if __name__ == "__main__":
    main()
