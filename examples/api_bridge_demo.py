import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np

from core.api_bridge import APIBridge, initialize_api_bridge
from core.entry_exit_logic import EntryExitLogic

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Bridge Demo Script

This demonstrates how to properly initialize and use the API bridge
for testing the Schwabot mathematical pipeline with real market data.

Usage:
    python examples/api_bridge_demo.py

Requirements:
    - Set API keys in environment variables or modify the script
    - Install required dependencies: ccxt, numpy
"""


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchawbotAPIDemo:
    """
    Demonstration of the API bridge integrated with the mathematical pipeline.
    """

    def __init__():-> None:
        """Initialize the demo system."""
        self.api_bridge: APIBridge = None
        self.entry_exit_logic: EntryExitLogic = None
        self.trading_symbols = ["BTC/USDC", "ETH/USDC", "BTC/USDT"]

    async def initialize():-> None:
        """Initialize the API bridge and trading components."""
        logger.info("🚀 Initializing Schwabot API Bridge Demo...")

        # Get API credentials from environment variables
        coinmarketcap_api_key = os.getenv("COINMARKETCAP_API_KEY")
        coinbase_api_key = os.getenv("COINBASE_API_KEY")
        coinbase_api_secret = os.getenv("COINBASE_API_SECRET")

        # Initialize API bridge
        self.api_bridge = await initialize_api_bridge(
            coinmarketcap_api_key=coinmarketcap_api_key,
            coinbase_api_key=coinbase_api_key,
            coinbase_api_secret=coinbase_api_secret,
            sandbox=True,  # Use sandbox for testing
        )

        # Initialize entry/exit logic engine
        self.entry_exit_logic = EntryExitLogic(
            entry_threshold=0.42,
            exit_threshold=-0.38,
            risk_management_enabled=True,
            enable_ghost_overlay=True,
            enable_ferris_phase=True,
        )

        logger.info("✅ Initialization complete!")

    async def test_price_data_fetching():-> Dict[str, Any]:
        """Test price data fetching functionality."""
        logger.info("📊 Testing price data fetching...")

        price_results = {}

        for symbol in self.trading_symbols:
            try:
                price_data = await self.api_bridge.fetch_price_data(symbol)
                price_results[symbol] = price_data

                logger.info(
                    f"💰 {symbol}: ${price_data['price']:.2f} "
                    f"(Volume: {price_data.get('volume_24h', 0):.2f})"
                )

            except Exception as e:
                logger.error(f"❌ Failed to fetch price for {symbol}: {e}")
                price_results[symbol] = None

        return price_results

    async def test_order_book_fetching():-> Dict[str, Any]:
        """Test order book data fetching functionality."""
        logger.info("📈 Testing order book fetching...")

        order_book_results = {}

        for symbol in self.trading_symbols:
            try:
                order_book = await self.api_bridge.fetch_order_book(symbol, limit=10)
                order_book_results[symbol] = order_book

                if order_book and order_book.get("bids") and order_book.get("asks"):
                    best_bid = order_book["bids"][0][0] if order_book["bids"] else 0
                    best_ask = order_book["asks"][0][0] if order_book["asks"] else 0
                    spread = (
                        ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
                    )

                    logger.info(
                        f"📊 {symbol}: Bid ${best_bid:.2f}, Ask ${best_ask:.2f}, "
                        f"Spread {spread:.4f}%"
                    )

            except Exception as e:
                logger.error(f"❌ Failed to fetch order book for {symbol}: {e}")
                order_book_results[symbol] = None

        return order_book_results

    async def test_mathematical_integration():-> Dict[str, Any]:
        """Test integration with the mathematical pipeline."""
        logger.info("🧮 Testing mathematical pipeline integration...")

        math_results = {}

        for symbol in self.trading_symbols:
            try:
                # Fetch market data
                price_data = await self.api_bridge.fetch_price_data(symbol)
                order_book = await self.api_bridge.fetch_order_book(symbol, limit=20)

                if not price_data or not order_book:
                    continue

                # Create vectorized data from order book
                bids = np.array([bid[0] for bid in order_book.get("bids", [])])
                asks = np.array([ask[0] for ask in order_book.get("asks", [])])

                if len(bids) == 0 or len(asks) == 0:
                    continue

                # Combine into a vector for mathematical analysis
                vector = np.concatenate([bids, asks])

                # Generate mock ferris phase and ghost input for testing
                ferris_phase = np.random.uniform(0, 2 * np.pi)
                ghost_input = np.random.uniform(-1, 1)

                # Test entry signal computation
                entry_signal = self.entry_exit_logic.compute_entry_signal(
                    vector=vector,
                    ferris_phase=ferris_phase,
                    ghost_input=ghost_input,
                    metadata={"symbol": symbol, "price": price_data["price"]},
                )

                math_results[symbol] = {
                    "entry_signal": entry_signal,
                    "vector_size": len(vector),
                    "ferris_phase": ferris_phase,
                    "ghost_input": ghost_input,
                }
                logger.info(
                    f"🎯 {symbol}: Entry Signal = {entry_signal['should_enter']}, "
                    f"Strength = {entry_signal['signal_strength']:.4f}"
                )

            except Exception as e:
                logger.error(f"❌ Mathematical integration failed for {symbol}: {e}")
                math_results[symbol] = None

        return math_results

    async def run_performance_test():-> None:
        """Run a performance test of the API bridge."""
        logger.info("⚡ Running performance test...")

        start_time = time.time()
        test_count = 10

        for i in range(test_count):
            symbol = self.trading_symbols[i % len(self.trading_symbols)]
            await self.api_bridge.fetch_price_data(symbol)

        total_time = time.time() - start_time
        avg_time = total_time / test_count

        logger.info(
            f"📊 Performance Test Results: "
            f"{test_count} requests in {total_time:.2f}s "
            f"(Avg: {avg_time:.3f}s per request)"
        )

        # Display API statistics
        stats = self.api_bridge.get_api_performance_summary()
        logger.info(f"📈 API Statistics: {stats}")

    async def run_demo():-> None:
        """Run the complete API bridge demonstration."""
        try:
            await self.initialize()

            # Test basic functionality
            price_results = await self.test_price_data_fetching()
            order_book_results = await self.test_order_book_fetching()

            # Test mathematical integration
            math_results = await self.test_mathematical_integration()

            # Run performance test
            await self.run_performance_test()

            # Summary
            logger.info("🎉 Demo completed successfully!")
            logger.info(f"📊 Tested {len(self.trading_symbols)} trading pairs")
            logger.info(
                f"✅ Price data: {sum(1 for v in price_results.values() if v)}/{len(price_results)}"
            )
            logger.info(
                f"✅ Order books: {sum(1 for v in order_book_results.values() if v)}/{len(order_book_results)}"
            )
            logger.info(
                f"✅ Math integration: {sum(1 for v in math_results.values() if v)}/{len(math_results)}"
            )

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise


async def main():-> None:
    """Main entry point."""
    demo = SchawbotAPIDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🤖 Schwabot API Bridge Demo")
    print("=" * 50)
    print("This demo shows how to integrate the API bridge with")
    print("the Schwabot mathematical pipeline for testing.")
    print()
    print("Set these environment variables for full functionality:")
    print("- COINMARKETCAP_API_KEY (optional)")
    print("- COINBASE_API_KEY (optional)")
    print("- COINBASE_API_SECRET (optional)")
    print()

    asyncio.run(main())
