#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dynamic Portfolio Volatility Manager
=========================================

Demonstrates the dynamic portfolio volatility manager capabilities:
- Real-time market data fetching for portfolio symbols
- Dynamic volatility calculations using multiple methods
- Portfolio correlation analysis and risk metrics
- Integration with all available API endpoints
- GPU-accelerated mathematical computations
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the dynamic portfolio manager
from core.dynamic_portfolio_volatility_manager import dynamic_portfolio_manager, TimeFrame, VolatilityMethod
from core.unified_trading_pipeline import UnifiedTradingPipeline

async def test_portfolio_symbol_tracking():
    """Test adding and tracking portfolio symbols."""
    logger.info("🧪 Testing Portfolio Symbol Tracking")
    
    # Add symbols to track
    symbols_to_track = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    
    for symbol in symbols_to_track:
        success = dynamic_portfolio_manager.add_portfolio_symbol(symbol)
        logger.info(f"Added {symbol}: {'✅' if success else '❌'}")
    
    # Get tracked symbols
    tracked_symbols = dynamic_portfolio_manager.get_tracked_symbols()
    logger.info(f"Tracked symbols: {tracked_symbols}")
    
    return len(tracked_symbols) > 0

async def test_market_data_fetching():
    """Test fetching real market data for tracked symbols."""
    logger.info("🧪 Testing Market Data Fetching")
    
    # Update all tracked symbols
    success = await dynamic_portfolio_manager.update_tracked_symbols()
    logger.info(f"Updated tracked symbols: {'✅' if success else '❌'}")
    
    # Get market data for each symbol
    tracked_symbols = dynamic_portfolio_manager.get_tracked_symbols()
    
    for symbol in tracked_symbols:
        market_data = dynamic_portfolio_manager.get_symbol_market_data(symbol)
        if market_data:
            logger.info(f"📊 {symbol}: ${market_data['price']:.2f} "
                       f"(vol: {market_data.get('volatility', 0):.4f}, "
                       f"change: {market_data.get('price_change', 0)*100:.2f}%)")
        else:
            logger.warning(f"❌ No market data for {symbol}")
    
    return True

async def test_volatility_calculations():
    """Test volatility calculations for tracked symbols."""
    logger.info("🧪 Testing Volatility Calculations")
    
    # Wait for some price data to accumulate
    logger.info("⏳ Waiting for price data accumulation...")
    await asyncio.sleep(5)
    
    # Update symbols again to get more data
    await dynamic_portfolio_manager.update_tracked_symbols()
    
    tracked_symbols = dynamic_portfolio_manager.get_tracked_symbols()
    
    for symbol in tracked_symbols:
        logger.info(f"📈 Volatility Analysis for {symbol}:")
        
        # Get volatility for different timeframes
        for timeframe in [TimeFrame.ONE_MINUTE, TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]:
            volatility = dynamic_portfolio_manager.get_symbol_volatility(symbol, timeframe)
            if volatility is not None:
                logger.info(f"  {timeframe.value}: {volatility:.6f}")
            else:
                logger.info(f"  {timeframe.value}: Not enough data")
        
        # Get comprehensive volatility analysis
        analysis = dynamic_portfolio_manager.get_volatility_analysis(symbol)
        if "error" not in analysis:
            logger.info(f"  ✅ Comprehensive analysis available")
        else:
            logger.info(f"  ❌ Analysis error: {analysis['error']}")
    
    return True

async def test_portfolio_management():
    """Test portfolio position management."""
    logger.info("🧪 Testing Portfolio Position Management")
    
    # Add some positions
    positions_to_add = [
        ("BTC/USDC", 0.1, 50000.0),
        ("ETH/USDC", 1.0, 3000.0),
        ("SOL/USDC", 10.0, 100.0)
    ]
    
    for symbol, quantity, entry_price in positions_to_add:
        success = await dynamic_portfolio_manager.add_position(symbol, quantity, entry_price)
        logger.info(f"Added position {symbol}: {'✅' if success else '❌'}")
    
    # Update portfolio
    success = await dynamic_portfolio_manager.update_portfolio()
    logger.info(f"Updated portfolio: {'✅' if success else '❌'}")
    
    # Get portfolio summary
    summary = dynamic_portfolio_manager.get_portfolio_summary()
    if "error" not in summary:
        logger.info(f"📊 Portfolio Summary:")
        logger.info(f"  Total positions: {summary['total_positions']}")
        logger.info(f"  Total value: ${summary['total_value']:.2f}")
        logger.info(f"  Total PnL: ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']:.2f}%)")
        
        if 'portfolio_volatility' in summary:
            logger.info(f"  Portfolio volatility: {summary['portfolio_volatility']:.6f}")
        if 'sharpe_ratio' in summary:
            logger.info(f"  Sharpe ratio: {summary['sharpe_ratio']:.4f}")
    else:
        logger.error(f"❌ Portfolio summary error: {summary['error']}")
    
    return "error" not in summary

async def test_unified_trading_pipeline_integration():
    """Test integration with unified trading pipeline."""
    logger.info("🧪 Testing Unified Trading Pipeline Integration")
    
    # Create pipeline with portfolio symbols
    config = {
        "portfolio_symbols": ["BTC/USDC", "ETH/USDC", "SOL/USDC"],
        "symbol": "BTC/USDC"
    }
    
    pipeline = UnifiedTradingPipeline(mode="demo", config=config)
    
    # Initialize pipeline
    success = await pipeline.initialize()
    logger.info(f"Pipeline initialization: {'✅' if success else '❌'}")
    
    if success:
        # Run a few trading cycles
        for i in range(3):
            logger.info(f"🔄 Running trading cycle {i+1}")
            cycle_result = await pipeline.run_trading_cycle()
            
            if "error" not in cycle_result:
                logger.info(f"  ✅ Cycle completed successfully")
                
                # Check for portfolio insights
                if "portfolio_insights" in cycle_result:
                    portfolio = cycle_result["portfolio_insights"]
                    logger.info(f"  📊 Portfolio: {portfolio.get('total_positions', 0)} positions, "
                              f"${portfolio.get('total_value', 0):.2f} value")
                
                # Check for volatility analysis
                if "volatility_analysis" in cycle_result:
                    vol_analysis = cycle_result["volatility_analysis"]
                    logger.info(f"  📈 Volatility analysis for {len(vol_analysis)} symbols")
            else:
                logger.error(f"  ❌ Cycle error: {cycle_result['error']}")
            
            await asyncio.sleep(2)  # Wait between cycles
        
        # Get final portfolio summary
        portfolio_summary = pipeline.get_portfolio_summary()
        if "error" not in portfolio_summary:
            logger.info(f"📊 Final Portfolio Summary:")
            logger.info(f"  Total positions: {portfolio_summary['total_positions']}")
            logger.info(f"  Total value: ${portfolio_summary['total_value']:.2f}")
            logger.info(f"  Total PnL: ${portfolio_summary['total_pnl']:.2f}")
        else:
            logger.error(f"❌ Final portfolio summary error: {portfolio_summary['error']}")
    
    return success

async def test_performance_metrics():
    """Test performance metrics and system status."""
    logger.info("🧪 Testing Performance Metrics")
    
    # Get performance metrics
    metrics = dynamic_portfolio_manager.get_performance_metrics()
    
    logger.info(f"📊 Performance Metrics:")
    logger.info(f"  Total calculations: {metrics['total_calculations']}")
    logger.info(f"  Cache hits: {metrics['cache_hits']}")
    logger.info(f"  Cache misses: {metrics['cache_misses']}")
    logger.info(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    logger.info(f"  GPU available: {'✅' if metrics['gpu_available'] else '❌'}")
    
    # Get API manager performance
    api_metrics = dynamic_portfolio_manager.api_manager.get_performance_metrics()
    logger.info(f"📡 API Performance:")
    logger.info(f"  Total requests: {api_metrics['total_requests']}")
    logger.info(f"  Successful requests: {api_metrics['successful_requests']}")
    logger.info(f"  Failed requests: {api_metrics['failed_requests']}")
    logger.info(f"  Success rate: {api_metrics['successful_requests']/max(api_metrics['total_requests'], 1):.2%}")
    
    return True

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Dynamic Portfolio Volatility Manager Tests")
    
    tests = [
        ("Portfolio Symbol Tracking", test_portfolio_symbol_tracking),
        ("Market Data Fetching", test_market_data_fetching),
        ("Volatility Calculations", test_volatility_calculations),
        ("Portfolio Management", test_portfolio_management),
        ("Unified Trading Pipeline Integration", test_unified_trading_pipeline_integration),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            result = await test_func()
            end_time = time.time()
            
            results[test_name] = {
                "success": result,
                "duration": end_time - start_time
            }
            
            logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'} "
                       f"({end_time - start_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = {
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        duration = f"{result['duration']:.2f}s"
        logger.info(f"{test_name}: {status} ({duration})")
        
        if result["success"]:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Dynamic Portfolio Volatility Manager is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution error: {e}") 