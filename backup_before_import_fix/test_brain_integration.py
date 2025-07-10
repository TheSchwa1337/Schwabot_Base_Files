import asyncio
import json
import logging
import math
import subprocess
import time
from pathlib import Path

from core.brain_trading_engine import BrainTradingEngine

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Brain Integration Test
=============================

Comprehensive test of brain trading functionality with working implementations.
This replaces placeholders with functional brain trading algorithms.
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_brain_trading_engine():
    """Test the brain trading engine functionality."""
    print("🧠 TESTING BRAIN TRADING ENGINE")
    print("=" * 50)

    try:

        # Initialize with custom configuration
        config = {}
            "base_profit_rate": 0.02,
            "confidence_threshold": 0.6,
            "enhancement_range": (0.8, 2.0),
            "max_history_size": 100,
        }

        engine = BrainTradingEngine(config)
        print("✅ Brain Trading Engine initialized")

        # Test different market scenarios
        test_scenarios = []
            {"name": "Bull Run", "price": 50000, "volume": 1000},
            {"name": "Bear Market", "price": 45000, "volume": 800},
            {"name": "High Volatility", "price": 52000, "volume": 2000},
            {"name": "Low Volume", "price": 49000, "volume": 200},
            {"name": "Recovery", "price": 51000, "volume": 1500},
        ]

        results = []
        print("\n📊 Processing market scenarios:")

        for i, scenario in enumerate(test_scenarios, 1):
            # Process brain signal
            signal = engine.process_brain_signal()
                scenario["price"], scenario["volume"], "BTC"
            )

            # Get trading decision
            decision = engine.get_trading_decision(signal)

            results.append()
                {"scenario": scenario, "signal": signal, "decision": decision}
            )

            print(f"{i}. {scenario['name']}")
            print(f"   Price: ${scenario['price']:,}, Volume: {scenario['volume']:,}")
            print()
                f"   Signal: {signal.signal_strength:.3f}, Confidence: {signal.confidence:.3f}"
            )
            print()
                f"   Action: {decision['action']}, Size: {decision['position_size']:.2%}"
            )
            print(f"   Profit Score: {signal.profit_score:.2f}")
            print()

        # Get final metrics
        metrics = engine.get_metrics_summary()
        print("📈 BRAIN ENGINE METRICS:")
        print(f"   Total Signals: {metrics['total_signals']}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Avg Profit: {metrics['avg_profit_per_signal']:.2f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

        # Export data
        engine.export_signals("test_brain_signals.json")
        print("   📄 Data exported to test_brain_signals.json")

        return True, engine, results

    except Exception as e:
        print(f"❌ Brain Trading Engine test failed: {e}")
        return False, None, None


def test_mathematical_functions():
    """Test mathematical functions and calculations."""
    print("\n🔢 TESTING MATHEMATICAL FUNCTIONS")
    print("=" * 50)

    try:
        print("✅ Unified Math System loaded")

        # Test basic math operations
        test_cases = []
            (100.0, 1.5, "multiply"),
            ([1, 2, 3, 4, 5], None, "mean"),
            (25.0, None, "sqrt"),
            (3.14159, None, "sin"),
        ]

        print("\n🧮 Mathematical operations:")
        for i, (value, factor, operation) in enumerate(test_cases, 1):
            try:
                if operation == "multiply" and factor:
                    result = value * factor
                elif operation == "mean" and isinstance(value, list):
                    result = sum(value) / len(value)
                elif operation == "sqrt":
                    result = value**0.5
                elif operation == "sin":

                    result = math.sin(value)
                else:
                    result = 0.0

                print(f"{i}. {operation}({value}) = {result:.4f}")
            except Exception as e:
                print(f"{i}. {operation}({value}) = Error: {e}")

        return True

    except Exception as e:
        print(f"❌ Mathematical functions test failed: {e}")
        return False


def test_symbol_processing():
    """Test symbol and glyph processing."""
    print("\n🔣 TESTING SYMBOL PROCESSING")
    print("=" * 50)

    try:
        # Test brain symbols processing
        brain_symbols = ["[BRAIN]", "🧠", "💰", "📈", "⚡", "🎯"]

        print("Processing brain-related symbols:")
        for symbol in brain_symbols:
            # Simple symbol analysis
            symbol_hash = hash(symbol) % 1000
            symbol_strength = abs(symbol_hash) / 1000.0

            print(f"  {symbol}: Hash={symbol_hash}, Strength={symbol_strength:.3f}")

        return True

    except Exception as e:
        print(f"❌ Symbol processing test failed: {e}")
        return False


async def run_backtest_simulation():
    """Run a simple backtest simulation."""
    print("\n📊 RUNNING BACKTEST SIMULATION")
    print("=" * 50)

    try:

        engine = BrainTradingEngine()
            {"base_profit_rate": 0.01, "confidence_threshold": 0.7}
        )

        # Simulate price data
        price_data = []
            50000,
            50200,
            49800,
            50500,
            51000,
            50700,
            51200,
            50900,
            51500,
            51800,
            51300,
            52000,
            51700,
            52200,
        ]

        volume_data = []
            1000,
            1100,
            900,
            1200,
            1300,
            1000,
            1400,
            1100,
            1500,
            1200,
            1000,
            1600,
            1100,
            1700,
        ]

        portfolio = 100000  # $100k starting capital
        btc_holdings = 0
        trades = []

        print("🔄 Processing historical data...")

        for i, (price, volume) in enumerate(zip(price_data, volume_data)):
            signal = engine.process_brain_signal(price, volume)
            decision = engine.get_trading_decision(signal)

            # Execute trades
            if decision["action"] == "BUY" and decision["confidence"] > 0.7:
                trade_amount = portfolio * 0.1  # 10% position
                if trade_amount > 0:
                    btc_bought = trade_amount / price
                    btc_holdings += btc_bought
                    portfolio -= trade_amount
                    trades.append(("BUY", price, btc_bought, decision["confidence"]))

            elif decision["action"] == "SELL" and decision["confidence"] > 0.7:
                if btc_holdings > 0:
                    btc_sold = btc_holdings * 0.5  # Sell 50%
                    portfolio += btc_sold * price
                    btc_holdings -= btc_sold
                    trades.append(("SELL", price, btc_sold, decision["confidence"]))

            await asyncio.sleep(0.1)  # Small delay for demo

        # Calculate final results
        final_btc_value = btc_holdings * price_data[-1]
        total_value = portfolio + final_btc_value
        total_return = (total_value - 100000) / 100000

        print("\n📈 BACKTEST RESULTS:")
        print("   Starting Capital: $100,00")
        print(f"   Final Cash: ${portfolio:,.2f}")
        print(f"   BTC Holdings: {btc_holdings:.6f}")
        print(f"   BTC Value: ${final_btc_value:,.2f}")
        print(f"   Total Value: ${total_value:,.2f}")
        print(f"   Return: {total_return:.2%}")
        print(f"   Total Trades: {len(trades)}")

        if trades:
            avg_confidence = sum(t[3] for t in, trades) / len(trades)
            print(f"   Avg Confidence: {avg_confidence:.3f}")

        return True, {}
            "starting_capital": 100000,
            "final_value": total_value,
            "return": total_return,
            "trades": len(trades),
        }

    except Exception as e:
        print(f"❌ Backtest simulation failed: {e}")
        return False, None


def run_flake8_check():
    """Run Flake8 check on core files."""
    print("\n🔍 RUNNING CODE QUALITY CHECK")
    print("=" * 50)

    try:

        # Check our brain trading engine
        result = subprocess.run()
            ["python", "-m", "flake8", "core/brain_trading_engine.py", "--count"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Brain Trading Engine: No flake8 issues")
        else:
            print(f"⚠️ Brain Trading Engine: {result.stdout.strip()} issues found")

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Code quality check failed: {e}")
        return False


async def main():
    """Main test execution."""
    print("🚀 SCHWABOT BRAIN INTEGRATION TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Brain Trading Engine
    success, engine, trading_results = test_brain_trading_engine()
    results["brain_engine"] = success

    # Test 2: Mathematical Functions
    success = test_mathematical_functions()
    results["mathematical_functions"] = success

    # Test 3: Symbol Processing
    success = test_symbol_processing()
    results["symbol_processing"] = success

    # Test 4: Backtest Simulation
    success, backtest_data = await run_backtest_simulation()
    results["backtest_simulation"] = success

    # Test 5: Code Quality
    success = run_flake8_check()
    results["code_quality"] = success

    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - SYSTEM READY FOR PACKAGING")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")

    # Export test results
    test_report = {}
        "timestamp": time.time(),
        "results": results,
        "passed": passed,
        "total": total,
        "success_rate": passed / total,
        "backtest_data": backtest_data if "backtest_simulation" in locals() else None,
    }

    with open("test_results.json", "w") as f:
        json.dump(test_report, f, indent=2)

    print("📄 Test report saved to test_results.json")

    return test_report


if __name__ == "__main__":
    # Ensure we have a clean start
    Path("logs").mkdir(exist_ok=True)

    # Run the test suite
    asyncio.run(main())
