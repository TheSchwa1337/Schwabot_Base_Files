#!/usr/bin/env python3
"""
Test script for ZPE-ZBE integration into clean trading pipeline.
"""

import asyncio
import logging
from typing import Any, Dict

    CleanTradingPipeline, MarketData, TradingAction, StrategyBranch
)
from core.zpe_zbe_core import QuantumSyncStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_zpe_zbe_integration():
    """Test the ZPE-ZBE integration in the clean trading pipeline."""

    logger.info("🧪 Testing ZPE-ZBE Integration in Clean Trading Pipeline")
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

    # Test enhanced market data processing
    logger.info("📊 Testing enhanced market data processing...")
    zpe_zbe_market_data = pipeline._enhance_market_data_with_zpe_zbe(market_data)

    logger.info(f"   ZPE Energy: {zpe_zbe_market_data.zpe_vector.energy:.2e}")
    logger.info(f"   ZBE Status: {zpe_zbe_market_data.zbe_balance.status:.3f}")
    logger.info(f"   Quantum Sync Status: {zpe_zbe_market_data.quantum_sync_status.value}")
    logger.info(f"   Quantum Potential: {zpe_zbe_market_data.quantum_potential:.3f}")
    logger.info(f"   Strategy Confidence: {zpe_zbe_market_data.strategy_confidence:.3f}")

    # Test enhanced strategy selection
    logger.info("🎯 Testing enhanced strategy selection...")
    regime = pipeline._analyze_market_regime(market_data)
    strategy, zpe_zbe_analysis = pipeline._enhance_strategy_selection_with_zpe_zbe(market_data, regime)

    logger.info(f"   Selected Strategy: {strategy.value}")
    logger.info(f"   ZPE Analysis: {zpe_zbe_analysis}")

    # Test enhanced risk management
    logger.info("🛡️ Testing enhanced risk management...")
    test_signal = {}
        'action': 'BUY',
        'quantity': 0.1,
        'price': 50000.0,
        'confidence': 0.8
    }

    enhanced_signal = pipeline._enhance_risk_management_with_zpe_zbe(test_signal, zpe_zbe_market_data)
    logger.info(f"   Enhanced Signal: {enhanced_signal}")

    # Test pipeline summary
    logger.info("📈 Testing enhanced pipeline summary...")
    summary = pipeline.get_zpe_zbe_pipeline_summary()

    logger.info("   ZPE-ZBE Metrics:")
    for key, value in summary['zpe_zbe_metrics'].items():
        logger.info(f"     {key}: {value}")

    logger.info("   Quantum Performance:")
    for key, value in summary['quantum_performance'].items():
        logger.info(f"     {key}: {value}")

    logger.info("✅ ZPE-ZBE integration test completed successfully!")


if __name__ == '__main__':
    import time
    asyncio.run(test_zpe_zbe_integration())
