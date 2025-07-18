#!/usr/bin/env python3
"""
Test script for CRLF integration into clean trading pipeline.
"""

import asyncio
import logging
from typing import Any, Dict

import numpy as np

    CleanTradingPipeline, MarketData, TradingAction, StrategyBranch
)
from core.chrono_recursive_logic_function import CRLFTriggerState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_crlf_integration():
    """Test the CRLF integration in the clean trading pipeline."""

    logger.info("🧪 Testing CRLF Integration in Clean Trading Pipeline")
    logger.info("=" * 60)

    # Create pipeline
    pipeline = CleanTradingPipeline()
        symbol="BTCUSDT",
        initial_capital=10000.0
    )

    # Create test market data
    market_data = MarketData()
        symbol="BTCUSDT",
        price=50000.0,
        volume=1000.0,
        timestamp=time.time(),
        bid=49999.0,
        ask=50001.0,
        volatility=0.3,
        trend_strength=0.7,
        entropy_level=3.5
    )

    # Test CRLF enhanced market data processing
    logger.info("📊 Testing CRLF enhanced market data processing...")
    crlf_market_data = pipeline._enhance_market_data_with_crlf(market_data)

    logger.info(f"   CRLF Output: {crlf_market_data.crlf_response.crlf_output:.4f}")
    logger.info(f"   Trigger State: {crlf_market_data.trigger_state.value}")
    logger.info(f"   Strategy Alignment: {crlf_market_data.strategy_alignment_score:.3f}")
    logger.info(f"   Temporal Resonance: {crlf_market_data.temporal_resonance:.3f}")
    logger.info(f"   Recursion Depth: {crlf_market_data.recursion_depth}")

    # Test CRLF performance summary
    logger.info("📈 Testing CRLF performance summary...")
    crlf_summary = pipeline.get_crlf_performance_summary()

    logger.info("   CRLF Performance:")
    for key, value in crlf_summary['crlf_performance'].items():
        logger.info(f"     {key}: {value}")

    logger.info("   Pipeline CRLF State:")
    for key, value in crlf_summary['pipeline_crlf_state'].items():
        logger.info(f"     {key}: {value}")

    # Test multiple iterations to see CRLF evolution
    logger.info("🔄 Testing CRLF evolution over multiple iterations...")
    for i in range(5):
        # Update market data with slight variations
        market_data.price += np.random.normal(0, 100)
        market_data.volatility = np.clip(market_data.volatility + np.random.normal(0, 0.1), 0.0, 1.0)
        market_data.trend_strength = np.clip(market_data.trend_strength + np.random.normal(0, 0.1), 0.0, 1.0)

        crlf_market_data = pipeline._enhance_market_data_with_crlf(market_data)

        logger.info(f"   Iteration {i+1}: CRLF={crlf_market_data.crlf_response.crlf_output:.4f}, ")
                   f"State={crlf_market_data.trigger_state.value}, "
                   f"Depth={crlf_market_data.recursion_depth}")

    logger.info("✅ CRLF integration test completed successfully!")


if __name__ == '__main__':
    import time
    asyncio.run(test_crlf_integration())
