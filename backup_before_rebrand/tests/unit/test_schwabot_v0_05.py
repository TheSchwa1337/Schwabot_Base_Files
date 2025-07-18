import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np

from core.fallback_logic import FallbackLogic, FallbackType
from core.ferris_rde import FerrisPhase, FerrisRDE
from core.fractal_core import FractalCore, FractalState, FractalType
from core.glyph_vm import GlyphState, GlyphType, GlyphVM
from core.matrix_fault_resolver import MatrixFaultResolver
from core.matrix_map_logic import LogicHashType, MatrixMapLogic, MatrixType
from core.profit_cycle_allocator import ProfitCycleAllocator
from core.strategy_mapper import StrategyMapper, StrategyType
from core.wallet_tracker import AssetType, PositionType, WalletTracker

#!/usr/bin/env python3
"""
Schwabot v0.5 Test Suite
=========================

Comprehensive test suite for Schwabot v0.5 modules.
Tests all core functionality and integration.
"""


# Add schwabot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "schwabot"))

# Import all Schwabot modules

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchwabotV05Tester:
    """Comprehensive tester for Schwabot v0.5."""

    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.start_time = time.time()

        # Initialize all modules
        self.strategy_mapper = StrategyMapper()
        self.ferris_rde = FerrisRDE()
        self.profit_allocator = ProfitCycleAllocator()
        self.wallet_tracker = WalletTracker()
        self.fallback_logic = FallbackLogic()
        self.glyph_vm = GlyphVM()
        self.matrix_logic = MatrixMapLogic()
        self.fractal_core = FractalCore()
        self.fault_resolver = MatrixFaultResolver()

        logger.info("🚀 Schwabot v0.5 Test Suite initialized")

    def run_all_tests():-> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive Schwabot v0.5 tests...")

        tests = []
            ("Strategy Mapper", self.test_strategy_mapper),
            ("Ferris RDE", self.test_ferris_rde),
            ("Profit Cycle Allocator", self.test_profit_allocator),
            ("Wallet Tracker", self.test_wallet_tracker),
            ("Fallback Logic", self.test_fallback_logic),
            ("Glyph VM", self.test_glyph_vm),
            ("Matrix Map Logic", self.test_matrix_logic),
            ("Fractal Core", self.test_fractal_core),
            ("Matrix Fault Resolver", self.test_fault_resolver),
            ("Integration Tests", self.test_integration),
        ]
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name} tests...")
                result = test_func()
                self.test_results[test_name] = result
                logger.info(f"✅ {test_name} tests completed: {result['status']}")
            except Exception as e:
                logger.error(f"❌ {test_name} tests failed: {e}")
                self.test_results[test_name] = {}
                    "status": "FAILED",
                    "error": str(e),
                    "details": {},
                }
        # Generate summary
        total_tests = len(tests)
        passed_tests = sum()
            1
            for result in self.test_results.values()
            if result.get("status") == "PASSED"
        )

        summary = {}
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "test_results": self.test_results,
            "execution_time": time.time() - self.start_time,
        }
        logger.info(f"🎯 Test Summary: {passed_tests}/{total_tests} tests passed")
        return summary

    def test_strategy_mapper():-> Dict[str, Any]:
        """Test strategy mapper functionality."""
        try:
            # Test strategy creation
            strategy_id = self.strategy_mapper.create_strategy()
                "test_strategy", StrategyType.HASH_BASED, {"param1": 0.5}
            )
            assert strategy_id is not None

            # Test strategy activation
            success = self.strategy_mapper.activate_strategy(strategy_id)
            assert success

            # Test strategy selection
            selected = self.strategy_mapper.select_strategy()
                {"market_condition": "bullish"}
            )
            assert selected is not None

            # Test strategy routing
            route = self.strategy_mapper.route_strategy(strategy_id, {"signal": "buy"})
            assert route is not None

            return {}
                "status": "PASSED",
                "details": {}
                    "strategies_created": 1,
                    "strategies_activated": 1,
                    "strategies_selected": 1,
                    "routes_generated": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_ferris_rde():-> Dict[str, Any]:
        """Test Ferris RDE functionality."""
        try:
            # Test cycle creation
            cycle = self.ferris_rde.start_cycle("test_cycle")
            assert cycle is not None
            assert cycle.phase == FerrisPhase.TICK

            # Test phase updates
            market_data = {"price": 50000, "volume": 1000000, "volatility": 0.2}
            phase = self.ferris_rde.update_phase(market_data)
            assert phase in []
                FerrisPhase.TICK,
                FerrisPhase.PIVOT,
                FerrisPhase.ASCENT,
                FerrisPhase.DESCENT,
            ]

            # Test signal generation
            signal = self.ferris_rde.generate_signal(market_data)
            # Signal might be None if conditions aren't met'
            if signal:
                assert signal.signal_type in []
                    "buy",
                    "sell",
                    "hold",
                    "scale_in",
                    "scale_out",
                ]

            # Test cycle completion
            completed_cycle = self.ferris_rde.end_cycle()
            assert completed_cycle is not None

            return {}
                "status": "PASSED",
                "details": {}
                    "cycles_created": 1,
                    "phases_updated": 1,
                    "signals_generated": 1 if signal else 0,
                    "cycles_completed": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_profit_allocator():-> Dict[str, Any]:
        """Test profit cycle allocator functionality."""
        try:
            # Test cycle creation
            cycle = self.profit_allocator.start_profit_cycle("test_profit_cycle")
            assert cycle is not None

            # Test profit allocation
            portfolio_state = {"total_value": 10000, "volatility": 0.1}
            allocations = self.profit_allocator.allocate_profit(1000.0, portfolio_state)
            assert len(allocations) > 0

            # Test cycle completion
            completed_cycle = self.profit_allocator.end_cycle()
            assert completed_cycle is not None
            assert completed_cycle.total_profit == 1000.0

            return {}
                "status": "PASSED",
                "details": {}
                    "cycles_created": 1,
                    "allocations_made": len(allocations),
                    "total_profit_allocated": 1000.0,
                    "cycles_completed": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_wallet_tracker():-> Dict[str, Any]:
        """Test wallet tracker functionality."""
        try:
            # Test position creation
            position = self.wallet_tracker.add_position()
                AssetType.BTC, PositionType.LONG, 0.1, 50000.0
            )
            assert position is not None

            # Test position update
            success = self.wallet_tracker.update_position_price()
                position.position_id, 51000.0
            )
            assert success

            # Test transaction creation
            transaction = self.wallet_tracker.add_transaction()
                AssetType.ETH, "buy", 1.0, 3000.0, 10.0
            )
            assert transaction is not None

            # Test snapshot creation
            snapshot = self.wallet_tracker.create_snapshot()
            assert snapshot is not None

            return {}
                "status": "PASSED",
                "details": {}
                    "positions_created": 1,
                    "positions_updated": 1,
                    "transactions_created": 1,
                    "snapshots_created": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_fallback_logic():-> Dict[str, Any]:
        """Test fallback logic functionality."""
        try:
            # Test stall detection
            system_state = {"last_activity": time.time() - 400, "performance": 0.3}
            stall_detector = self.fallback_logic.check_for_stall(system_state)
            # Should detect stall due to time threshold

            # Test fallback triggering
            if stall_detector:
                event = self.fallback_logic.trigger_fallback()
                    FallbackType.RE_ENTRY, "System stalled", system_state
                )
                assert event is not None

            # Test failure recording
            self.fallback_logic.record_failure("Test failure")
            assert self.fallback_logic.consecutive_failures == 1

            # Test success recording
            self.fallback_logic.record_success()
            assert self.fallback_logic.consecutive_failures == 0

            return {}
                "status": "PASSED",
                "details": {}
                    "stall_detections": 1 if stall_detector else 0,
                    "fallbacks_triggered": 1 if stall_detector else 0,
                    "failures_recorded": 1,
                    "successes_recorded": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_glyph_vm():-> Dict[str, Any]:
        """Test glyph VM functionality."""
        try:
            # Test glyph creation
            glyph = self.glyph_vm.add_glyph()
                "test_glyph", GlyphType.SYSTEM, GlyphState.ACTIVE, 0.8
            )
            assert glyph is not None

            # Test glyph update
            success = self.glyph_vm.update_glyph("test_glyph", 0.9)
            assert success

            # Test pattern detection
            patterns = self.glyph_vm.detect_patterns()
            # Patterns might be empty initially

            # Test display rendering
            display = self.glyph_vm.render_display()
            assert len(display) > 0

            return {}
                "status": "PASSED",
                "details": {}
                    "glyphs_created": 1,
                    "glyphs_updated": 1,
                    "patterns_detected": len(patterns),
                    "display_rendered": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_matrix_logic():-> Dict[str, Any]:
        """Test matrix map logic functionality."""
        try:
            # Test matrix creation
            matrix_data = np.random.rand(5, 5)
            matrix = self.matrix_logic.add_matrix()
                "test_matrix", MatrixType.FEATURE, matrix_data
            )
            assert matrix is not None

            # Test similarity calculation
            matrix2_data = np.random.rand(5, 5)
            self.matrix_logic.add_matrix()
                "test_matrix2", MatrixType.FEATURE, matrix2_data
            )
            similarity = self.matrix_logic.calculate_similarity()
                "test_matrix", "test_matrix2"
            )
            assert 0.0 <= similarity <= 1.0

            # Test logic hash creation
            hash_obj = self.matrix_logic.create_logic_hash()
                LogicHashType.STRATEGY, "test_matrix", matrix_data, 0.8
            )
            assert hash_obj is not None

            # Test hash selection
            selected_hash = self.matrix_logic.select_logic_hash("test_matrix")
            # Might be None if no similar matrices

            return {}
                "status": "PASSED",
                "details": {}
                    "matrices_created": 2,
                    "similarities_calculated": 1,
                    "hashes_created": 1,
                    "hashes_selected": 1 if selected_hash else 0,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_fractal_core():-> Dict[str, Any]:
        """Test fractal core functionality."""
        try:
            # Test fractal creation
            fractal_data = np.random.rand(100)
            fractal = self.fractal_core.add_fractal()
                "test_fractal", FractalType.PRICE, FractalState.ACTIVE, fractal_data
            )
            assert fractal is not None

            # Test fractal update
            new_data = np.random.rand(100)
            success = self.fractal_core.update_fractal("test_fractal", new_data)
            assert success

            # Test pattern detection
            patterns = self.fractal_core.detect_patterns()
            # Patterns might be empty initially

            return {}
                "status": "PASSED",
                "details": {}
                    "fractals_created": 1,
                    "fractals_updated": 1,
                    "patterns_detected": len(patterns),
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_fault_resolver():-> Dict[str, Any]:
        """Test matrix fault resolver functionality."""
        try:
            # Test matrix health analysis
            matrix = np.random.rand(5, 5)
            health = self.fault_resolver.analyze_matrix_health("test_matrix", matrix)
            assert health is not None
            assert 0.0 <= health.health_score <= 1.0

            # Test fault detection (create a problematic, matrix)
            bad_matrix = np.array([[1, 1], [1, 1.000001]])  # Nearly singular
            health_bad = self.fault_resolver.analyze_matrix_health()
                "bad_matrix", bad_matrix
            )
            assert health_bad is not None

            # Test fault resolution
            if len(self.fault_resolver.faults) > 0:
                fault_id = list(self.fault_resolver.faults.keys())[0]
                self.fault_resolver.resolve_fault(fault_id, bad_matrix)
                # Resolution might be None if no improvement

            return {}
                "status": "PASSED",
                "details": {}
                    "health_analyses": 2,
                    "faults_detected": len(self.fault_resolver.faults),
                    "resolutions_attempted": 1
                    if len(self.fault_resolver.faults) > 0
                    else 0,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def test_integration():-> Dict[str, Any]:
        """Test integration between modules."""
        try:
            # Simulate a complete trading cycle
            market_data = {}
                "price": 50000,
                "volume": 1000000,
                "volatility": 0.2,
                "market_condition": "bullish",
            }
            # 1. Strategy selection
            strategy_id = self.strategy_mapper.create_strategy()
                "integration_strategy", StrategyType.HASH_BASED, {}
            )
            self.strategy_mapper.activate_strategy(strategy_id)
            self.strategy_mapper.select_strategy(market_data)

            # 2. Ferris cycle
            self.ferris_rde.start_cycle("integration_cycle")
            self.ferris_rde.update_phase(market_data)
            signal = self.ferris_rde.generate_signal(market_data)

            # 3. Wallet tracking
            if signal and signal.signal_type == "buy":
                self.wallet_tracker.add_position()
                    AssetType.BTC, PositionType.LONG, 0.1, market_data["price"]
                )
                self.wallet_tracker.add_transaction()
                    AssetType.BTC, "buy", 0.1, market_data["price"], 5.0
                )

            # 4. Profit allocation
            self.profit_allocator.start_profit_cycle("integration_profit")
            portfolio_state = {"total_value": 10000, "volatility": 0.1}
            self.profit_allocator.allocate_profit(500.0, portfolio_state)

            # 5. Glyph updates
            self.glyph_vm.update_glyph("trading_performance", 0.7)
            self.glyph_vm.update_glyph("strategy_confidence", 0.8)

            # 6. Matrix updates
            matrix_data = np.random.rand(10, 10)
            self.matrix_logic.update_matrix("market_features", matrix_data)

            # 7. Fractal updates
            fractal_data = np.random.rand(100)
            self.fractal_core.update_fractal("price_fractal", fractal_data)

            # 8. Health monitoring
            self.fault_resolver.analyze_matrix_health("market_features", matrix_data)

            return {}
                "status": "PASSED",
                "details": {}
                    "strategy_integration": 1,
                    "ferris_integration": 1,
                    "wallet_integration": 1 if signal else 0,
                    "profit_integration": 1,
                    "glyph_integration": 1,
                    "matrix_integration": 1,
                    "fractal_integration": 1,
                    "fault_integration": 1,
                },
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e), "details": {}}

    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("🎯 SCHWABOT v0.5 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Execution Time: {summary['execution_time']:.2f}s")
        print("\nDetailed Results:")
        print("-" * 40)

        for test_name, result in summary["test_results"].items():
            status = result.get("status", "UNKNOWN")
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")

            if status == "FAILED" and "error" in result:
                print(f"   Error: {result['error']}")

        print("\n" + "=" * 60)

        if summary["success_rate"] >= 0.8:
            print("🎉 Schwabot v0.5 is ready for deployment!")
        else:
            print("⚠️  Some tests failed. Please review and fix issues.")


def main():
    """Main test execution."""
    print("🚀 Starting Schwabot v0.5 Test Suite...")

    tester = SchwabotV05Tester()
    summary = tester.run_all_tests()
    tester.print_summary(summary)

    # Exit with appropriate code
    if summary["success_rate"] >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
