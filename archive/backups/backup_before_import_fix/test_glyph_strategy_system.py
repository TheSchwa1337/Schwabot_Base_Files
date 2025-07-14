import logging
import os
import sys
import traceback

from core.basket_vector_linker import BasketVectorLinker
from core.glyph_phase_resolver import GlyphPhaseResolver
from core.profit_memory_echo import ProfitMemoryEcho
from core.quantum_superpositional_trigger import QuantumSuperpositionalTrigger
from core.strategy import create_glyph_trading_system
from core.strategy.entry_exit_portal import EntryExitPortal  # Moved to top-level
from core.strategy.glyph_gate_engine import GlyphGateEngine
from core.strategy.glyph_strategy_core import GlyphStrategyCore
from core.strategy.zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate
from core.warp_sync_core import WarpSyncCore
from drawdown_predictor import DrawdownPredictor  # Assuming it's in the root'

# -*- coding: utf-8 -*-
"""
Test Script for Glyph Strategy System
------------------------------------
Comprehensive test and demonstration of the glyph-to-strategy mapping system
    for Schwabot's mathematical trading framework.'

This script demonstrates:
1. Glyph to strategy bit mapping via SHA256
2. Gear-driven strategy selection based on volume
3. Fractal memory encoding and storage
4. Entry/exit portal integration
5. Simulated trade execution
"""


# Add the parent directory to sys.path to allow imports from 'core'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the newly created modules for testing

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_glyph_strategy_core():
    """Test the core glyph strategy functionality."""
    print("=" * 60)
    print("GLYPH STRATEGY CORE TEST")
    print("=" * 60)

    try:
        # Initialize core
        core = GlyphStrategyCore()
            enable_fractal_memory=True,
            enable_gear_shifting=True,
            volume_thresholds=(1.5e6, 5e6),
        )

        # Test glyphs
        test_glyphs = []
            "brain",
            "skull",
            "fire",
            "hourglass",
            "tornado",
            "lightning",
            "shield",
            "target",
            "crystal",
            "scales",
        ]
        test_volumes = [1e6, 3e6, 6e6]  # Low, medium, high volume

        print()
            f"Testing {len(test_glyphs)} glyphs across {len(test_volumes)} volume levels"
        )
        print()

        results = []

        for glyph in test_glyphs:
            print(f"Glyph: {glyph}")
            glyph_results = []

            for volume in test_volumes:
                # Get strategy selection
                result = core.select_strategy(glyph, volume)

                glyph_results.append()
                    {}
                        "volume": volume,
                        "gear_state": result.gear_state,
                        "strategy_id": result.strategy_id,
                        "confidence": result.confidence,
                        "fractal_hash": result.fractal_hash[:8] + "...",
                    }
                )

                print()
                    f"  Volume: {volume:.1e} -> Gear: {result.gear_state}-bit, "
                    f"Strategy: {result.strategy_id}, Confidence: {result.confidence:.3f}"
                )

            results.append({"glyph": glyph, "results": glyph_results})
            print()

        # Show performance stats
        print("PERFORMANCE STATISTICS")
        print("-" * 40)
        stats = core.get_performance_stats()
        for key, value in stats.items():
            if key == "fractal_memory":
                print("Fractal Memory:")
                for mem_key, mem_value in value.items():
                    print(f"  {mem_key}: {mem_value}")
            else:
                print(f"{key}: {value}")

        return core, results

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, []
    except Exception as e:
        print(f"X Test failed: {e}")
        print(traceback.format_exc())
        return None, []


def test_entry_exit_portal():
    """Test the entry/exit portal functionality."""
    print("\n" + "=" * 60)
    print("ENTRY/EXIT PORTAL TEST")
    print("=" * 60)

    try:
        # Initialize portal
        portal = EntryExitPortal()
            enable_risk_management=True,
            enable_portfolio_tracking=True,
            max_position_size=0.1,
            min_confidence_threshold=0.5,
        )

        # Test parameters
        test_glyphs = ["brain", "skull", "fire", "hourglass", "tornado"]
        test_volume = 3.2e6
        test_price = 50000.0
        test_asset = "BTC/USD"

        print("Testing trade signal processing with:")
        print(f"  Volume: {test_volume:.1e}")
        print(f"  Price: ${test_price:,.2f}")
        print(f"  Asset: {test_asset}")
        print()

        executed_trades = []

        for glyph in test_glyphs:
            print(f"Processing glyph: {glyph}")

            # Process signal
            signal = portal.process_glyph_signal()
                glyph, test_volume, test_asset, test_price
            )

            if signal:
                print("  Signal generated:")
                print(f"    Strategy ID: {signal.strategy_id}")
                print(f"    Direction: {signal.direction.value}")
                print(f"    Confidence: {signal.confidence:.3f}")
                print(f"    Gear State: {signal.metadata.get('gear_state', 'N/A')}")

                # Execute signal (simulated)
                result = portal.execute_signal(signal, dry_run=True)

                if "execution_result" in result:
                    exec_result = result["execution_result"]
                    print("  Execution result:")
                    print(f"    Status: {exec_result['status']}")
                    print(f"    Order ID: {exec_result['order_id']}")
                    print(f"    Size: ${exec_result['executed_size']:,.2f}")
                    print(f"    Fees: ${exec_result['fees']:,.2f}")

                    executed_trades.append()
                        {"glyph": glyph, "signal": signal, "execution": exec_result}
                    )
                else:
                    print()
                        f"  X Execution failed: {result.get('error', 'Unknown error')}"
                    )
            else:
                print("  X Signal rejected (confidence too, low)")

            print()

        # Show portal stats
        print("PORTAL STATISTICS")
        print("-" * 40)
        stats = portal.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        return portal, executed_trades

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, []
    except Exception as e:
        print(f"X Test failed: {e}")
        print(traceback.format_exc())
        return None, []


def test_integrated_workflow():
    """Test the complete integrated workflow."""
    print("\n" + "=" * 60)
    print("INTEGRATED WORKFLOW TEST")
    print("=" * 60)

    try:

        # Create complete system
        glyph_core, portal = create_glyph_trading_system()
            enable_fractal_memory=True,
            enable_gear_shifting=True,
            enable_risk_management=True,
            enable_portfolio_tracking=True,
        )

        print("Complete glyph trading system created")
        print()

        # Simulate market conditions
        market_scenarios = []
            {"volume": 1e6, "price": 45000, "description": "Low volume, bearish"},
            {"volume": 3e6, "price": 50000, "description": "Medium volume, neutral"},
            {"volume": 7e6, "price": 55000, "description": "High volume, bullish"},
        ]

        test_glyphs = ["brain", "skull", "fire"]

        for scenario in market_scenarios:
            print(f"Market Scenario: {scenario['description']}")
            print()
                f"   Volume: {scenario['volume']:.1e}, Price: ${scenario['price']:,.2f}"
            )
            print()

            for glyph in test_glyphs:
                # Process signal
                signal = portal.process_glyph_signal()
                    glyph, scenario["volume"], "BTC/USD", scenario["price"]
                )

                if signal:
                    # Execute signal
                    result = portal.execute_signal(signal, dry_run=True)

                    print()
                        f"  {glyph} -> {signal.direction.value} "
                        f"(Strategy: {signal.strategy_id}, ")
                        f"Confidence: {signal.confidence:.3f})"
                    )

                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        print(f"    Status: {exec_result['status']}")
                        print(f"    Order ID: {exec_result['order_id']}")
                        print(f"    Size: ${exec_result['executed_size']:,.2f}")
                        print(f"    Fees: ${exec_result['fees']:,.2f}")

                        # Update portfolio (simplified)
                        if exec_result["status"] == "filled":
                            print(f"      Trade executed successfully for {glyph}.")
                        else:
                            print()
                                f"      Trade failed for {glyph}: {exec_result.get('error', 'Unknown error')}"
                            )
                else:
                    print(f"  {glyph} -> Signal rejected (confidence too, low).")
            print()

        # Print combined system statistics
        print("COMBINED SYSTEM STATISTICS")
        print("-" * 40)
        print("Glyph Core:")
        for k, v in glyph_core.get_performance_stats().items():
            if k == "fractal_memory":
                print("  Fractal Memory:")
                for mem_k, mem_v in v.items():
                    print(f"    {mem_k}: {mem_v}")
            else:
                print(f"  {k}: {v}")
        print("\nPortal:")
        for k, v in portal.get_performance_stats().items():
            if k == "portfolio_summary":
                print("  Portfolio Summary:")
                for port_k, port_v in v.items():
                    if isinstance(port_v, (float, int)):
                        print(f"    {port_k}: {port_v:,.2f}")
                    else:
                        print(f"    {port_k}: {port_v}")
            else:
                print(f"  {k}: {v}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"X Test failed: {e}")
        print(traceback.format_exc())
        return False


def test_glyph_gate_engine():
    """Test the Glyph Gate Engine functionality."""
    print("\n" + "=" * 60)
    print("GLYPH GATE ENGINE TEST")
    print("=" * 60)

    try:
        # Initialize individual components for the engine

        glyph_core_test, _ = create_glyph_trading_system()
            enable_fractal_memory=True, enable_gear_shifting=True
        )
        zygot_zalgo_gate_test = ZygotZalgoEntropyDualKeyGate()
        warp_sync_core_test = WarpSyncCore()
        quantum_trigger_test = QuantumSuperpositionalTrigger()
        # Need to provide initial strategies for BasketVectorLinker
        initial_strategies_for_linker_test = {}
            "TrendFollowing_EMA": [0.1, 0.2, 0.7, 0.5, 0.3],
            "MeanReversion_RSI": [0.8, 0.1, 0.5, 0.6, 0.1],
        }
        basket_linker_test = BasketVectorLinker(initial_strategies_for_linker_test)
        phase_resolver_test = GlyphPhaseResolver()
        profit_echo_test = ProfitMemoryEcho()

        engine = GlyphGateEngine()
            glyph_core=glyph_core_test,
            zygot_zalgo_gate=zygot_zalgo_gate_test,
            warp_sync_core=warp_sync_core_test,
            quantum_trigger=quantum_trigger_test,
            basket_linker=basket_linker_test,
            phase_resolver=phase_resolver_test,
            profit_echo=profit_echo_test,
            confidence_threshold=0.6,
        )

        # Simulate a series of market ticks
        market_ticks = []
            {}
                "glyph": "brain",
                "volume": 1.2e6,
                "price": 48000.0,
                "tick_id": 1,
                "internal_data": {"cpu_alignment": 0.8, "mem_usage": 0.5},
                "external_data": {"market_volatility": 0.6, "news_sentiment": 0.7},
            },
            {}
                "glyph": "skull",
                "volume": 3.5e6,
                "price": 50500.0,
                "tick_id": 2,
                "internal_data": {"cpu_alignment": 0.9, "mem_usage": 0.4},
                "external_data": {"market_volatility": 0.4, "news_sentiment": 0.8},
            },
        ]

        success = True
        for tick in market_ticks:
            print()
                f"\n--- Evaluating Signal for Glyph: {tick['glyph']}, Tick: {tick['tick_id']} ---"
            )
            decision = engine.evaluate_signal()
                glyph=tick["glyph"],
                volume_signal=tick["volume"],
                current_price=tick["price"],
                tick_id=tick["tick_id"],
                internal_system_data=tick["internal_data"],
                external_api_data=tick["external_data"],
                performance_feedback={"recent_profit": 0.1, "recent_loss": 0.05},
            )
            print()
                f"Final Decision: Gate Open = {decision.gate_open}, Reason = {decision.reason}, Confidence = {decision.confidence_score:.3f}"
            )
            if not decision.gate_open:
                success = False

        print("\n--- Decision History --- ")
        history = engine.get_decision_history()
        for dec in history:
            print()
                f"  Signal ID: {dec.signal_id}, Open: {dec.gate_open}, Conf: {dec.confidence_score:.3f}"
            )

        print("\n--- Resetting Engine ---")
        engine.reset_engine()
        print(f"Decision history after reset: {engine.get_decision_history()}")

        return success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"X Test failed: {e}")
        print(traceback.format_exc())
        return False


def test_drawdown_predictor():
    """Test the Drawdown Predictor functionality."""
    print("\n" + "=" * 60)
    print("DRAWDOWN PREDICTOR TEST")
    print("=" * 60)

    try:
        predictor = DrawdownPredictor(lookback_period=10, confidence_level=0.9)

        # Simulate PnL data over time
        simulated_pnl_data = []
            0.1,
            0.2,
            -0.05,
            0.15,
            -0.1,
            0.3,
            0.05,
            -0.25,
            0.1,
            0.0,  # Initial 10 for lookback
            -0.3,
            0.2,
            0.1,
            -0.15,
            0.05,
            -0.4,
            0.2,
            -0.1,
            0.15,
            0.0,  # Additional data
        ]

        print(f"Predictor initialized with lookback_period={predictor.lookback_period}")

        success = True
        print("\n--- Updating Historical Data and Predicting Drawdowns ---")
        for i, pnl in enumerate(simulated_pnl_data):
            predictor.update_historical_data(pnl)
            print(f"Step {i + 1}: Updated with PnL = {pnl:.3f}")

            prediction = predictor.predict_drawdown()
            if prediction:
                print(f"  Predicted Drawdown: {prediction['predicted_drawdown']:.4f}")
                print()
                    f"  Prediction Interval: ({prediction['lower_bound']:.4f}, {prediction['upper_bound']:.4f})"
                )
            else:
                print("  Prediction: Not enough data (expected for early, steps).")
                if ()
                    i >= predictor.lookback_period - 1
                ):  # After enough data, it should predict
                    success = False
                    print("❌ Prediction failed after sufficient data.")

        print("\n--- Final Metrics ---")
        metrics = predictor.get_metrics()
        for k, v in metrics.items():
            if isinstance(v, tuple):
                print(f"  {k}: ({v[0]:.4f}, {v[1]:.4f})")
            else:
                print(f"  {k}: {v}")

        if ()
            metrics["total_predictions"]
            < len(simulated_pnl_data) - predictor.lookback_period + 1
        ):
            success = False
            print("❌ Not all expected predictions were made.")

        print("\n--- Resetting Predictor ---")
        predictor.reset()
        print(f"Metrics after reset: {predictor.get_metrics()}")

        return success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"X Test failed: {e}")
        print(traceback.format_exc())
        return False


def demonstrate_mathematical_framework():
    """Demonstrate the mathematical framework behind the glyph system."""
    print("\n" + "=" * 60)
    print("MATHEMATICAL FRAMEWORK DEMONSTRATION")
    print("=" * 60)

    try:
        # GlyphStrategyCore is imported at the top-level
        core = GlyphStrategyCore()

        # Demonstrate SHA256 transformation
        print("SHA-256 Transformation Process:")
        test_glyph = "brain"
        sha_hash = core.glyph_to_sha(test_glyph)
        print(f"  Glyph: {test_glyph}")
        print(f"  SHA-256: {sha_hash}")
        print()

        # Demonstrate bit extraction
        print("Bit Extraction Process:")
        for bit_depth in [4, 8, 16]:
            strategy_bits = core.sha_to_strategy_bits(sha_hash, bit_depth)
            binary = bin(strategy_bits)[2:].zfill(bit_depth)
            print(f"  {bit_depth}-bit: {strategy_bits} (binary: {binary})")
        print()

        # Demonstrate gear shifting
        print("Gear Shifting Logic:")
        volumes = [1e6, 2e6, 4e6, 6e6]
        for volume in volumes:
            gear = core.gear_shift(volume)
            print(f"  Volume: {volume:.1e} -> Gear: {gear}-bit")
        print()

        # Demonstrate fractal memory
        print("Fractal Memory Encoding:")
        for i, glyph in enumerate(["brain", "skull", "fire"]):
            result = core.select_strategy(glyph, 3e6)
            print(f"  {glyph} -> Hash: {result.fractal_hash[:16]}...")

        fractal_stats = core.get_fractal_memory_stats()
        print(f"\n  Total hashes stored: {fractal_stats['total_hashes']}")
        print(f"  Memory size limit: {fractal_stats['memory_size']}")

        return True

    except ImportError as e:
        print(f"❌ Demonstration failed: {e}")
        return False
    except Exception as e:
        print(f"X Demonstration failed: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Main function to run all tests and demonstrations."""
    test_results = {}

    # Test 1: Glyph Strategy Core
    core_instance, _ = test_glyph_strategy_core()
    test_results["glyph_core"] = core_instance is not None

    # Test 2: Entry/Exit Portal
    portal_instance, _ = test_entry_exit_portal()
    test_results["entry_exit_portal"] = portal_instance is not None

    # Test 3: Integrated workflow
    workflow_result = test_integrated_workflow()
    test_results["workflow"] = workflow_result

    # Test 4: Mathematical framework
    math_result = demonstrate_mathematical_framework()
    test_results["mathematical"] = math_result

    # Test 5: Glyph Gate Engine
    gate_engine_result = test_glyph_gate_engine()
    test_results["glyph_gate_engine"] = gate_engine_result

    # Test 6: Drawdown Predictor
    drawdown_predictor_result = test_drawdown_predictor()
    test_results["drawdown_predictor"] = drawdown_predictor_result

    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    overall_status = all(test_results.values())
    final_message = ()
        "All tests passed successfully!"
        if overall_status
        else "Some tests failed. Please review the output above."
    )
    print("\n" + final_message)

    return overall_status


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
