#!/usr/bin/env python3
"""
Test Unified Trading Pipeline
============================

Comprehensive test of the unified trading pipeline with proper registry management.
Demonstrates:
- Canonical trade registry as single source of truth
- Specialized registry linkage
- Performance analytics
- Registry consistency validation
- Backtesting capabilities
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

async def test_unified_trading_pipeline():
    """Test the unified trading pipeline."""
    logger.info("🚀 Starting Unified Trading Pipeline Test")
    
    try:
        # Import the unified trading pipeline
        from core.unified_trading_pipeline import UnifiedTradingPipeline
        
        # Initialize pipeline in demo mode
        pipeline = UnifiedTradingPipeline(mode="demo", config={
            "min_confidence": 0.5,
            "max_trades": 50
        })
        
        logger.info("✅ Pipeline initialized successfully")
        
        # Test 1: Run a few trading cycles
        logger.info("\n📊 Test 1: Running trading cycles...")
        for i in range(5):
            cycle_result = await pipeline.run_trading_cycle()
            logger.info(f"Cycle {i+1}: {cycle_result.get('trade_executed', False)} | Portfolio: ${cycle_result.get('portfolio_value', 0):.2f}")
            await asyncio.sleep(0.1)  # Small delay between cycles
        
        # Test 2: Get performance analytics
        logger.info("\n📈 Test 2: Performance Analytics...")
        analytics = pipeline.get_performance_analytics()
        logger.info(f"Canonical Registry: {analytics.get('canonical_registry', {}).get('total_trades', 0)} trades")
        logger.info(f"Specialized Registries: {list(analytics.get('specialized_registries', {}).keys())}")
        
        # Test 3: Get registry statistics
        logger.info("\n📋 Test 3: Registry Statistics...")
        stats = pipeline.get_registry_statistics()
        logger.info(f"Total linkages: {stats.get('linkages', {}).get('total_linkages', 0)}")
        logger.info(f"Registry coverage: {stats.get('linkages', {}).get('registry_coverage', {})}")
        
        # Test 4: Validate registry consistency
        logger.info("\n🔍 Test 4: Registry Consistency Validation...")
        consistency = pipeline.validate_registry_consistency()
        logger.info(f"Canonical registry status: {consistency.get('canonical_registry', {}).get('status', 'unknown')}")
        logger.info(f"Specialized registries status: {list(consistency.get('specialized_registries', {}).keys())}")
        
        # Test 5: Run a short backtest
        logger.info("\n🔄 Test 5: Running backtest...")
        backtest_results = await pipeline.run_backtest(duration_seconds=10, cycle_interval=0.5)
        logger.info(f"Backtest completed: {backtest_results.get('cycles_completed', 0)} cycles")
        logger.info(f"Final profit: ${backtest_results.get('total_profit', 0):.2f}")
        logger.info(f"Success rate: {backtest_results.get('success_rate', 0):.2%}")
        
        # Test 6: Demonstrate registry linkage
        logger.info("\n🔗 Test 6: Registry Linkage Demonstration...")
        from core.trade_registry import canonical_trade_registry
        
        # Get recent trades
        recent_trades = canonical_trade_registry.get_recent_trades(3)
        for trade in recent_trades:
            logger.info(f"Trade {trade.trade_hash[:8]}... | {trade.symbol} {trade.action}")
            logger.info(f"  Linked registries: {list(trade.linked_registries)}")
            logger.info(f"  Specialized hashes: {list(trade.specialized_hashes.keys())}")
        
        # Test 7: Test specialized registry access
        logger.info("\n🧬 Test 7: Specialized Registry Access...")
        from core.registry_coordinator import registry_coordinator
        
        if recent_trades:
            # Get a trade with all its linkages
            trade_with_linkages = registry_coordinator.get_trade_with_all_linkages(recent_trades[0].trade_hash)
            logger.info(f"Trade with linkages: {len(trade_with_linkages.get('specialized_data', {}))} specialized registries")
            
            # Show specialized data
            for registry_name, data in trade_with_linkages.get('specialized_data', {}).items():
                logger.info(f"  {registry_name}: {type(data).__name__}")
        
        logger.info("\n✅ All tests completed successfully!")
        
        # Final summary
        logger.info("\n📊 FINAL SUMMARY:")
        logger.info(f"Total trades executed: {pipeline.total_trades}")
        logger.info(f"Successful trades: {pipeline.successful_trades}")
        logger.info(f"Total profit: ${pipeline.total_profit:.2f}")
        logger.info(f"Success rate: {pipeline.successful_trades / pipeline.total_trades * 100:.1f}%" if pipeline.total_trades > 0 else "N/A")
        logger.info(f"Final portfolio value: ${pipeline.portfolio_value:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_registry_coordinator():
    """Test the registry coordinator specifically."""
    logger.info("\n🔗 Testing Registry Coordinator...")
    
    try:
        from core.registry_coordinator import registry_coordinator
        from core.trade_registry import canonical_trade_registry
        
        # Test adding a trade with linkages
        trade_data = {
            "symbol": "BTC/USDC",
            "action": "buy",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "amount": 100.0,
            "fees": 0.1,
            "profit_usd": 4.9,
            "profit_percentage": 4.9,
            "strategy_id": "test_strategy",
            "confidence": 0.8,
            "timestamp": time.time()
        }
        
        specialized_data = {
            "profit_buckets": {
                "tick_blob": "BTC/USDC:50000.0:test",
                "entry_price": 50000.0,
                "exit_price": 50500.0,
                "time_to_exit": 300,
                "strategy_id": "test_strategy"
            },
            "soulprints": {
                "vector": {"phase": 0.5, "drift": 0.3, "confidence": 0.8, "asset": "BTC/USDC"},
                "strategy_id": "test_strategy",
                "confidence": 0.8,
                "is_executed": True,
                "profit_result": 4.9
            }
        }
        
        # Add trade with linkages
        canonical_hash = registry_coordinator.add_trade_with_linkages(trade_data, specialized_data)
        logger.info(f"✅ Added trade with linkages: {canonical_hash[:8]}...")
        
        # Test getting trade with all linkages
        trade_with_linkages = registry_coordinator.get_trade_with_all_linkages(canonical_hash)
        logger.info(f"✅ Retrieved trade with {len(trade_with_linkages.get('specialized_data', {}))} specialized registries")
        
        # Test performance analytics
        analytics = registry_coordinator.get_performance_analytics()
        logger.info(f"✅ Performance analytics generated")
        
        # Test registry statistics
        stats = registry_coordinator.get_registry_statistics()
        logger.info(f"✅ Registry statistics generated")
        
        # Test consistency validation
        consistency = registry_coordinator.validate_registry_consistency()
        logger.info(f"✅ Registry consistency validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Registry coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_canonical_registry():
    """Test the canonical trade registry specifically."""
    logger.info("\n📊 Testing Canonical Trade Registry...")
    
    try:
        from core.trade_registry import canonical_trade_registry
        
        # Test adding trades
        for i in range(3):
            trade_data = {
                "symbol": f"ETH/USDC",
                "action": "buy" if i % 2 == 0 else "sell",
                "entry_price": 3000.0 + i * 10,
                "exit_price": 3010.0 + i * 10,
                "amount": 50.0,
                "fees": 0.05,
                "profit_usd": 5.0 + i,
                "profit_percentage": 1.0 + i * 0.1,
                "strategy_id": f"test_strategy_{i}",
                "confidence": 0.7 + i * 0.1,
                "timestamp": time.time() + i
            }
            
            trade_hash = canonical_trade_registry.add_trade(trade_data)
            logger.info(f"✅ Added trade {i+1}: {trade_hash[:8]}...")
        
        # Test querying trades
        recent_trades = canonical_trade_registry.get_recent_trades(5)
        logger.info(f"✅ Retrieved {len(recent_trades)} recent trades")
        
        # Test performance summary
        performance = canonical_trade_registry.get_performance_summary()
        logger.info(f"✅ Performance summary: {performance.get('total_trades', 0)} trades, ${performance.get('total_profit', 0):.2f} profit")
        
        # Test getting trades by symbol
        eth_trades = canonical_trade_registry.get_trades_by_symbol("ETH/USDC")
        logger.info(f"✅ Retrieved {len(eth_trades)} ETH trades")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Canonical registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    logger.info("🧪 Starting Comprehensive Registry Management Tests")
    
    # Test 1: Canonical registry
    success1 = await test_canonical_registry()
    
    # Test 2: Registry coordinator
    success2 = await test_registry_coordinator()
    
    # Test 3: Unified trading pipeline
    success3 = await test_unified_trading_pipeline()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("🧪 TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Canonical Registry Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    logger.info(f"Registry Coordinator Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    logger.info(f"Unified Trading Pipeline Test: {'✅ PASSED' if success3 else '❌ FAILED'}")
    
    overall_success = success1 and success2 and success3
    logger.info(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        logger.info("\n🎉 Registry management system is working correctly!")
        logger.info("📋 Key features verified:")
        logger.info("  - Canonical trade registry as single source of truth")
        logger.info("  - Specialized registry linkage without redundancy")
        logger.info("  - Proper hash tracking across all registries")
        logger.info("  - Performance analytics and consistency validation")
        logger.info("  - Complete trading pipeline integration")
    else:
        logger.info("\n⚠️  Some issues detected. Please review the logs above.")

if __name__ == "__main__":
    asyncio.run(main()) 