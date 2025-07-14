#!/usr/bin/env python3
"""
Unified Trading Pipeline - Schwabot Full Math Chain Orchestrator
==============================================================

This script ties together all core components:
- Market data feed (simulated or real)
- CleanUnifiedMathSystem
- ChronoResonanceWeatherMapper
- TemporalWarpEngine
- TradeRegistry
- CLI/visual hooks

Usage:
    python run_trading_pipeline.py --mode [demo|live|backtest]
"""

import argparse
import asyncio
import logging
import sys
import time
from typing import Any, Dict

from core.clean_unified_math import CleanUnifiedMathSystem
from core.chrono_resonance_weather_mapper import ChronoResonanceWeatherMapper
from core.temporal_warp_engine import TemporalWarpEngine
from core.cli_live_entry import SchwabotCLI
from core.trade_registry import TradeRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedTradingPipeline")

class UnifiedTradingPipeline:
    def __init__(self, mode: str = "demo"):
        self.mode = mode
        self.math_system = CleanUnifiedMathSystem()
        self.weather_mapper = ChronoResonanceWeatherMapper()
        self.temporal_engine = TemporalWarpEngine()
        self.registry = TradeRegistry()
        self.cli = SchwabotCLI()
        self.active = False
        self.current_symbol = "BTC/USDT"
        self.portfolio_value = 10000.0
        self.last_price = 50000.0
        self.price_history = [self.last_price]
        self.step = 0

    async def start(self):
        logger.info(f"Starting Unified Trading Pipeline in {self.mode} mode...")
        self.active = True
        await self.cli.initialize_system(self.mode)
        while self.active:
            await self._step()
            await asyncio.sleep(1.0)

    async def _step(self):
        self.step += 1
        # Simulate market data
        price = self.last_price + self.math_system.mean([0, 0, 0, 0, 0]) + self.math_system.multiply(100, self.math_system.sin(time.time()))
        self.last_price = price
        self.price_history.append(price)
        if len(self.price_history) > 1000:
            self.price_history.pop(0)
        # Math chain
        base_profit = self.math_system.multiply(price, 0.01)
        enhancement = self.math_system.mean([0.5, 0.6, 0.7])
        confidence = self.math_system.mean([0.7, 0.8, 0.9])
        optimized_profit = self.math_system.optimize_profit(base_profit, enhancement, confidence)
        risk_adjusted = self.math_system.calculate_risk_adjustment(optimized_profit, 0.1, confidence)
        portfolio_weight = self.math_system.calculate_portfolio_weight(confidence, 0.2)
        # Chrono resonance
        crwf = self.weather_mapper.compute_crwf(time.time(), 40.0, -74.0, 100.0)
        # Temporal warp
        projected_time = self.temporal_engine.calculate_temporal_projection(time.time(), 0.1)
        # Registry
        trade_data = {
            'step': self.step,
            'symbol': self.current_symbol,
            'price': price,
            'optimized_profit': optimized_profit,
            'risk_adjusted': risk_adjusted,
            'portfolio_weight': portfolio_weight,
            'crwf': crwf,
            'projected_time': projected_time,
            'timestamp': time.time(),
        }
        trade_hash = self.registry.add_trade(trade_data)
        # CLI/visual output
        logger.info(f"[Step {self.step}] Price: {price:.2f} | Profit: {optimized_profit:.2f} | CRWF: {crwf:.3f} | Hash: {trade_hash[:8]}...")
        logger.info(f"Registry size: {len(self.registry.trades)} | Portfolio value: {self.portfolio_value:.2f}")
        # Stop after 10 steps in demo
        if self.mode == "demo" and self.step >= 10:
            self.active = False
            logger.info("Demo complete. Final registry:")
            for trade in self.registry.get_recent_trades(10):
                logger.info(trade)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Trading Pipeline")
    parser.add_argument('--mode', choices=['demo', 'live', 'backtest'], default='demo', help='Run mode')
    args = parser.parse_args()
    pipeline = UnifiedTradingPipeline(mode=args.mode)
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.") 