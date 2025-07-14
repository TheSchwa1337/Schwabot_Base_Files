#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Trading Loop Test

This script tests the full trading system to identify critical functionality issues
and ensure the bot is ready for live trading.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_trading_executor():
    """Test the trading executor with realistic configuration."""
    logger.info("🚀 STARTING COMPREHENSIVE TRADING LOOP TEST")
    logger.info("=" * 60)
    
    try:
        # Import the trading executor
        from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
        
        # Realistic configuration for testing
        exchange_config = {
            'exchange': 'coinbase',
            'api_key': 'test_key',  # Use test keys for safety
            'secret': 'test_secret',
            'sandbox': True,  # Always use sandbox for testing
            'enableRateLimit': True
        }
        
        strategy_config = {
            'strategy_type': 'entropy_enhanced',
            'timeframe': '1m',
            'confidence_threshold': 0.6
        }
        
        entropy_config = {
            'entropy_threshold': 0.7,
            'signal_strength_min': 0.3,
            'timing_adjustment': 0.2
        }
        
        risk_config = {
            'risk_tolerance': 0.02,  # 2% risk tolerance
            'profit_target': 0.05,   # 5% profit target
            'stop_loss': 0.01,       # 1% stop loss
            'position_size': 0.01,   # 0.01 BTC base position
            'max_position_size': 0.05, # 0.05 BTC max position
            'max_position': 0.1,     # 0.1 BTC max total position
            'min_trade_interval': 30 # 30 seconds between trades
        }
        
        logger.info("📋 Configuration loaded successfully")
        logger.info(f"Exchange: {exchange_config['exchange']}")
        logger.info(f"Sandbox mode: {exchange_config['sandbox']}")
        rt = risk_config['risk_tolerance']
        if isinstance(rt, dict):
            logger.info(f"Risk tolerance: {rt}")
        elif isinstance(rt, float):
            logger.info(f"Risk tolerance: {rt:.1%}")
        else:
            logger.info(f"Risk tolerance: {rt}")
        logger.info(f"Position size: {risk_config['position_size']} BTC")
        
        # Create trading executor
        logger.info("🔄 Creating trading executor...")
        executor = EntropyEnhancedTradingExecutor(
            exchange_config=exchange_config,
            strategy_config=strategy_config,
            entropy_config=entropy_config,
            risk_config=risk_config
        )
        
        logger.info("✅ Trading executor created successfully")
        
        # Test single trading cycle
        logger.info("🔄 Testing single trading cycle...")
        start_time = time.time()
        
        result = await executor.execute_trading_cycle()
        
        cycle_time = time.time() - start_time
        logger.info(f"⏱️ Trading cycle completed in {cycle_time:.2f} seconds")
        
        # Analyze result
        logger.info("📊 Trading cycle result analysis:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Action: {result.action.value}")
        logger.info(f"  Quantity: {result.executed_quantity:.6f} BTC")
        logger.info(f"  Price: ${result.executed_price:,.2f}")
        logger.info(f"  Fees: ${result.fees:.4f}")
        
        if not result.success:
            logger.warning(f"  Reason: {result.metadata.get('reason', 'unknown')}")
            if 'error' in result.metadata:
                logger.error(f"  Error: {result.metadata['error']}")
        
        # Get performance summary
        logger.info("📈 Performance summary:")
        performance = executor.get_performance_summary()
        for key, value in performance.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test multiple cycles
        logger.info("🔄 Testing multiple trading cycles...")
        cycle_results = []
        
        for i in range(3):
            logger.info(f"  Cycle {i+1}/3...")
            try:
                result = await executor.execute_trading_cycle()
                cycle_results.append(result)
                
                if result.success:
                    logger.info(f"    ✅ Success: {result.action.value} {result.executed_quantity:.6f} BTC")
                else:
                    logger.info(f"    ⚠️ No trade: {result.metadata.get('reason', 'unknown')}")
                
                # Wait between cycles
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"    ❌ Cycle {i+1} failed: {e}")
                cycle_results.append(None)
        
        # Final performance analysis
        logger.info("📊 Final performance analysis:")
        final_performance = executor.get_performance_summary()
        
        total_trades = final_performance.get('total_trades', 0)
        successful_trades = final_performance.get('successful_trades', 0)
        total_profit = final_performance.get('total_profit', 0.0)
        
        logger.info(f"  Total trades: {total_trades}")
        logger.info(f"  Successful trades: {successful_trades}")
        if total_trades > 0:
            win_rate = successful_trades / total_trades
            logger.info(f"  Win rate: {win_rate:.1%}")
        logger.info(f"  Total profit: ${total_profit:.4f}")
        
        # Identify critical issues
        logger.info("🔍 Critical functionality assessment:")
        
        issues_found = []
        
        # Check if exchange connection works
        if not result.success and 'exchange' in str(result.metadata.get('error', '')).lower():
            issues_found.append("❌ Exchange connection failed - check API keys and network")
        
        # Check if market data collection works
        if not result.success and 'market' in str(result.metadata.get('error', '')).lower():
            issues_found.append("❌ Market data collection failed - check exchange API")
        
        # Check if entropy processing works
        if executor.performance_metrics.get('entropy_adjustments', 0) == 0:
            issues_found.append("⚠️ No entropy adjustments made - entropy processing may not be working")
        
        # Check if risk management is working
        if executor.performance_metrics.get('risk_blocks', 0) > 0:
            logger.info(f"  Risk management blocked {executor.performance_metrics['risk_blocks']} trades")
        
        # Check calculation performance
        avg_time = final_performance.get('average_time_ms', 0)
        if avg_time > 1000:  # More than 1 second
            issues_found.append(f"⚠️ Slow calculation time: {avg_time:.0f}ms - may need optimization")
        
        if not issues_found:
            logger.info("✅ No critical issues found - system appears ready for live trading")
        else:
            logger.warning("⚠️ Critical issues found:")
            for issue in issues_found:
                logger.warning(f"  {issue}")
        
        # Recommendations
        logger.info("💡 Recommendations for live trading:")
        logger.info("  1. Replace test API keys with real exchange API keys")
        logger.info("  2. Set sandbox=False for live trading")
        logger.info("  3. Start with small position sizes")
        logger.info("  4. Monitor performance metrics closely")
        logger.info("  5. Set up proper error handling and alerts")
        
        return {
            'success': True,
            'total_cycles': len(cycle_results),
            'successful_cycles': sum(1 for r in cycle_results if r and r.success),
            'total_profit': total_profit,
            'issues_found': issues_found,
            'performance': final_performance
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

async def test_btc_pipeline():
    """Test the BTC trading pipeline."""
    logger.info("🚀 TESTING BTC TRADING PIPELINE")
    logger.info("=" * 40)
    
    try:
        from core.unified_btc_trading_pipeline import create_btc_trading_pipeline
        
        # Create pipeline
        pipeline = create_btc_trading_pipeline()
        
        # Test with sample data
        test_prices = [50000, 50100, 50200, 50150, 50300]
        test_volumes = [1000000, 1200000, 1100000, 900000, 1300000]
        
        results = []
        for i, (price, volume) in enumerate(zip(test_prices, test_volumes)):
            logger.info(f"Processing price ${price:,.0f}, volume {volume:,.0f}")
            
            result = pipeline.process_btc_price(price, volume)
            results.append(result)
            
            if result.success and result.signal:
                signal = result.signal
                logger.info(f"  Signal: {signal.signal_type.upper()}")
                logger.info(f"  Amount: {signal.amount:.6f} BTC")
                logger.info(f"  Confidence: {signal.confidence:.3f}")
                logger.info(f"  Recommendation: {result.execution_recommendation}")
            else:
                logger.info(f"  No signal generated")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        logger.info(f"Pipeline summary: {summary}")
        
        return {
            'success': True,
            'signals_generated': sum(1 for r in results if r.signal),
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"❌ BTC pipeline test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

async def main():
    """Run all tests."""
    logger.info("🧪 COMPREHENSIVE TRADING SYSTEM TEST")
    logger.info("=" * 60)
    
    # Test 1: Trading Executor
    logger.info("\n📊 TEST 1: Trading Executor")
    executor_result = await test_trading_executor()
    
    # Test 2: BTC Pipeline
    logger.info("\n📊 TEST 2: BTC Trading Pipeline")
    pipeline_result = await test_btc_pipeline()
    
    # Summary
    logger.info("\n📋 TEST SUMMARY")
    logger.info("=" * 40)
    
    if executor_result['success']:
        logger.info("✅ Trading Executor: PASSED")
        logger.info(f"   Cycles: {executor_result['total_cycles']}")
        logger.info(f"   Successful: {executor_result['successful_cycles']}")
        logger.info(f"   Profit: ${executor_result['total_profit']:.4f}")
    else:
        logger.error("❌ Trading Executor: FAILED")
        logger.error(f"   Error: {executor_result['error']}")
    
    if pipeline_result['success']:
        logger.info("✅ BTC Pipeline: PASSED")
        logger.info(f"   Signals: {pipeline_result['signals_generated']}")
    else:
        logger.error("❌ BTC Pipeline: FAILED")
        logger.error(f"   Error: {pipeline_result['error']}")
    
    # Critical improvements needed
    logger.info("\n🎯 CRITICAL IMPROVEMENTS NEEDED")
    logger.info("=" * 40)
    
    if executor_result['success'] and executor_result.get('issues_found'):
        for issue in executor_result['issues_found']:
            logger.warning(f"  {issue}")
    
    logger.info("  1. Implement missing core components (PortfolioTracker, etc.)")
    logger.info("  2. Add proper error handling and recovery mechanisms")
    logger.info("  3. Implement real-time market data feeds")
    logger.info("  4. Add position management and P&L tracking")
    logger.info("  5. Implement proper risk management rules")
    logger.info("  6. Add logging and monitoring infrastructure")
    logger.info("  7. Test with real exchange APIs (sandbox first)")
    
    logger.info("\n🚀 Ready for next phase of development!")

if __name__ == "__main__":
    asyncio.run(main()) 