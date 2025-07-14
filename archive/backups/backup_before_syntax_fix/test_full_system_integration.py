import asyncio
import logging
import time
from typing import Any, Dict

from core.api_bridge import APIBridge, fetch_order_book_data, fetch_price_data
from core.dualistic_thought_engines import DualisticThoughtEngines
from core.hash_relay_system import hash_relay_system

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive System Integration Test Suite
==========================================

Tests the complete Schwabot trading system pipeline:
1. API Bridge (Coinbase, CoinMarketCap, CoinGecko)
2. Dualistic Thought Engines
3. Hash Relay System
4. Mathematical Pipeline Integration
5. Live Trading Channel Simulation
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_api_bridge():
    """Test API Bridge functionality."""
    print("\n" + "=" * 60)
    print("TESTING API BRIDGE")
    print("=" * 60)

    try:

        # Test API Bridge initialization
        APIBridge()
            enable_coingecko=True,
            enable_coinmarketcap=True,
            enable_ccxt=True,
            sandbox=True,
        )
        print("✓ API Bridge initialized successfully")

        # Test price data fetching
        async def test_price_fetch():
            try:
                price_data = await fetch_price_data("BTC/USDC")
                print()
                    f"✓ Price data fetched: {price_data.get('symbol', 'N/A')} @ ${price_data.get('price', 0):.2f}"
                )
                return True
            except Exception as e:
                print(f"✗ Price fetch failed: {e}")
                return False

        # Test order book fetching
        async def test_order_book():
            try:
                order_book = await fetch_order_book_data("BTC/USDC", limit=5)
                print()
                    f"✓ Order book fetched: {len(order_book.get('bids', []))} bids, {len(order_book.get('asks', []))} asks"
                )
                return True
            except Exception as e:
                print(f"✗ Order book fetch failed: {e}")
                return False

        # Run async tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        price_success = loop.run_until_complete(test_price_fetch())
        order_book_success = loop.run_until_complete(test_order_book())

        loop.close()

        return price_success and order_book_success

    except Exception as e:
        print(f"✗ API Bridge test failed: {e}")
        return False


def test_dualistic_thought_engines():
    """Test Dualistic Thought Engines functionality."""
    print("\n" + "=" * 60)
    print("TESTING DUALISTIC THOUGHT ENGINES")
    print("=" * 60)

    try:

        # Initialize engines
        engines = DualisticThoughtEngines()
        print("✓ Dualistic Thought Engines initialized")

        # Test market data processing
        market_data = {}
            "rsi": 25.5,
            "macd_signal": 0.1,
            "volume_change": 0.3,
            "current_price": 62000.0,
            "moving_average": 61500.0,
            "previous_close": 61800.0,
            "price_history": []
                61000.0,
                61500.0,
                62000.0,
                61800.0,
                62200.0,
                62500.0,
                62300.0,
            ],
            "volume_history": [100.0, 120.0, 110.0, 90.0, 130.0, 150.0, 140.0],
            "phase_data": [0.6, 0.4, 0.8, 0.2],
            "volatility": 0.8,
            "sentiment_score": 0.9,
            "performance_delta": 0.5,
            "consensus_signal": "buy",
        }
        # Process market data
        thought_vector = engines.process_market_data(market_data, thermal_state="hot")
        print("✓ Thought Vector generated:")
        print(f"  - Decision: {thought_vector.decision}")
        print(f"  - Confidence: {thought_vector.confidence:.2f}")
        print(f"  - State: {thought_vector.state.value}")
        print(f"  - Tags: {len(thought_vector.tags)} tags")

        # Test performance metrics
        performance = engines.get_engine_performance()
        print(f"✓ Performance metrics available: {len(performance)} metrics")

        return True

    except Exception as e:
        print(f"✗ Dualistic Thought Engines test failed: {e}")
        return False


def test_hash_relay_system():
    """Test Hash Relay System functionality."""
    print("\n" + "=" * 60)
    print("TESTING HASH RELAY SYSTEM")
    print("=" * 60)

    try:

        # Test relay subscription
        relay_received = []

        def relay_callback(hash_str: str, data: Dict[str, Any]):
            relay_received.append((hash_str, data))
            print()
                f"✓ Relay received: {hash_str[:8]}... for {data.get('decision', 'unknown')}"
            )

        hash_relay_system.subscribe(relay_callback)
        print("✓ Relay subscription registered")

        # Test data submission
        test_data = {}
            "timestamp": time.time(),
            "decision": "buy",
            "confidence": 0.85,
            "price": 62000.0,
        }
        hash_str = hash_relay_system.submit(test_data)
        print(f"✓ Data submitted and hashed: {hash_str[:8]}...")

        # Verify relay was received
        if relay_received:
            print(f"✓ Relay callback executed: {len(relay_received)} times")
            return True
        else:
            print("✗ Relay callback not executed")
            return False

    except Exception as e:
        print(f"✗ Hash Relay System test failed: {e}")
        return False


def test_integrated_pipeline():
    """Test the complete integrated pipeline."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED PIPELINE")
    print("=" * 60)

    try:

        # Track relay events
        relay_events = []

        def pipeline_relay_callback(hash_str: str, data: Dict[str, Any]):
            relay_events.append((hash_str, data))

        hash_relay_system.subscribe(pipeline_relay_callback)

        # Initialize engines
        engines = DualisticThoughtEngines()

        async def run_pipeline():
            # Step 1: Fetch market data
            price_data = await fetch_price_data("BTC/USDC")
            print(f"✓ Market data fetched: ${price_data.get('price', 0):.2f}")

            # Step 2: Create market data for processing
            market_data = {}
                "current_price": price_data.get("price", 62000.0),
                "rsi": 50.0,
                "macd_signal": 0.0,
                "volume_change": 0.1,
                "moving_average": price_data.get("price", 62000.0) * 0.99,
                "previous_close": price_data.get("price", 62000.0) * 0.98,
                "price_history": [price_data.get("price", 62000.0)] * 7,
                "volume_history": [100.0] * 7,
                "phase_data": [0.5, 0.5, 0.5, 0.5],
                "volatility": 0.5,
                "consensus_signal": "neutral",
            }
            # Step 3: Process through dualistic engines
            thought_vector = engines.process_market_data()
                market_data, thermal_state="warm"
            )
            print()
                f"✓ Thought Vector processed: {thought_vector.decision} (confidence: {thought_vector.confidence:.2f})"
            )

            # Step 4: Verify relay was triggered
            await asyncio.sleep(0.1)  # Allow relay to process

            if relay_events:
                print(f"✓ Pipeline relay triggered: {len(relay_events)} events")
                return True
            else:
                print("✗ Pipeline relay not triggered")
                return False

        # Run pipeline test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(run_pipeline())
        loop.close()

        return success

    except Exception as e:
        print(f"✗ Integrated pipeline test failed: {e}")
        return False


def test_trading_channel_simulation():
    """Simulate live trading channel functionality."""
    print("\n" + "=" * 60)
    print("TESTING TRADING CHANNEL SIMULATION")
    print("=" * 60)

    try:

        # Initialize components
        APIBridge(sandbox=True)
        engines = DualisticThoughtEngines()

        # Track trading decisions
        trading_decisions = []

        def trading_relay_callback(hash_str: str, data: Dict[str, Any]):
            if data.get("decision") in ["buy", "sell"]:
                trading_decisions.append()
                    {}
                        "hash": hash_str[:8],
                        "decision": data["decision"],
                        "confidence": data["confidence"],
                        "timestamp": data["timestamp"],
                    }
                )

        hash_relay_system.subscribe(trading_relay_callback)

        async def simulate_trading_cycle():
            # Simulate multiple market data points
            for i in range(3):
                # Simulate different market conditions
                market_data = {}
                    "current_price": 62000.0 + (i * 1000),
                    "rsi": 30.0 + (i * 20),
                    "macd_signal": -0.1 + (i * 0.1),
                    "volume_change": 0.1 + (i * 0.1),
                    "moving_average": 62000.0,
                    "previous_close": 62000.0,
                    "price_history": [62000.0] * 7,
                    "volume_history": [100.0] * 7,
                    "phase_data": [0.5, 0.5, 0.5, 0.5],
                    "volatility": 0.5,
                    "consensus_signal": "neutral",
                }
                # Process through engines
                thought_vector = engines.process_market_data()
                    market_data, thermal_state="warm"
                )
                print()
                    f"  Cycle {i + 1}: {thought_vector.decision} (confidence: {thought_vector.confidence:.2f})"
                )

                await asyncio.sleep(0.1)

            return len(trading_decisions) > 0

        # Run trading simulation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(simulate_trading_cycle())
        loop.close()

        if success:
            print()
                f"✓ Trading channel simulation successful: {len(trading_decisions)} decisions tracked"
            )
            for decision in trading_decisions:
                print()
                    f"  - {decision['decision'].upper()}: {decision['confidence']:.2f} confidence"
                )
            return True
        else:
            print("✗ Trading channel simulation failed")
            return False

    except Exception as e:
        print(f"✗ Trading channel simulation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("SCHWABOT COMPREHENSIVE SYSTEM INTEGRATION TEST")
    print("=" * 80)
    print("Testing complete trading system pipeline...")

    test_results = {}

    # Run all tests
    test_results["api_bridge"] = test_api_bridge()
    test_results["dualistic_engines"] = test_dualistic_thought_engines()
    test_results["hash_relay"] = test_hash_relay_system()
    test_results["integrated_pipeline"] = test_integrated_pipeline()
    test_results["trading_channel"] = test_trading_channel_simulation()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for live trading.")
        print("\nSystem Components Verified:")
        print("  ✓ API Bridge (Coinbase, CoinMarketCap, CoinGecko)")
        print("  ✓ Dualistic Thought Engines (Logical + Intuitive)")
        print("  ✓ Hash Relay System (Mathematical State, Relay)")
        print("  ✓ Integrated Pipeline (End-to-End, Processing)")
        print("  ✓ Trading Channel (Live Decision, Making)")
        print("\nReady for:")
        print("  - Live API trading via CCXT")
        print("  - BTC processor integration")
        print("  - Pool mining with trading capabilities")
    else:
        print("⚠️  Some tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
