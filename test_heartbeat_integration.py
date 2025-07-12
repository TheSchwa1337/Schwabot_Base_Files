#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Heartbeat Integration
==========================
Demonstrates how the Heartbeat Integration Manager coordinates all
advanced Schwabot modules with a 5-minute heartbeat cycle.

This script shows the integration of:
- Thermal Strategy Router
- Autonomic Limit Layer
- API Tick Cache
- Profit Echo Cache
- Drift Band Profiler
- GPU Logic Mapper
- Profit Projection Engine
"""

import asyncio
import logging
import time
from datetime import datetime

# Import the heartbeat integration manager
from core.heartbeat_integration_manager import HeartbeatIntegrationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_heartbeat_integration():
    """Test the heartbeat integration manager."""
    logger.info("🚀 Starting Heartbeat Integration Test")
    
    # Initialize the heartbeat integration manager
    heartbeat_manager = HeartbeatIntegrationManager()
    
    try:
        # Initialize the manager
        logger.info("🔄 Initializing Heartbeat Integration Manager...")
        if not await heartbeat_manager.initialize():
            logger.error("❌ Failed to initialize Heartbeat Integration Manager")
            return
        
        # Start the manager
        logger.info("🚀 Starting Heartbeat Integration Manager...")
        if not await heartbeat_manager.start():
            logger.error("❌ Failed to start Heartbeat Integration Manager")
            return
        
        # Run a few heartbeat cycles
        logger.info("💓 Running heartbeat cycles...")
        
        for cycle in range(3):  # Run 3 cycles for demonstration
            logger.info(f"\n{'='*60}")
            logger.info(f"💓 HEARTBEAT CYCLE {cycle + 1}")
            logger.info(f"{'='*60}")
            
            # Run a single heartbeat cycle
            cycle_result = await heartbeat_manager.run_heartbeat_cycle()
            
            # Display cycle results
            logger.info(f"📊 Cycle Status: {cycle_result['status']}")
            logger.info(f"⏱️  Execution Time: {cycle_result['execution_time']:.2f}s")
            logger.info(f"🎯 Strategies Processed: {cycle_result['strategies_processed']}")
            logger.info(f"🌡️  Thermal State: {cycle_result['thermal_state']}")
            logger.info(f"💾 Memory Usage: {cycle_result['memory_usage']:.3f}")
            logger.info(f"💰 Profit Echo Strength: {cycle_result['profit_echo_strength']:.3f}")
            
            if cycle_result['modules_processed']:
                logger.info(f"🔧 Modules Processed: {', '.join(cycle_result['modules_processed'])}")
            
            if cycle_result['warnings']:
                logger.warning(f"⚠️  Warnings: {cycle_result['warnings']}")
            
            if cycle_result['errors']:
                logger.error(f"❌ Errors: {cycle_result['errors']}")
            
            # Get integration stats
            stats = heartbeat_manager.get_integration_stats()
            logger.info(f"📈 Total Heartbeats: {stats['heartbeat_manager']['heartbeat_count']}")
            logger.info(f"✅ Success Rate: {stats['heartbeat_manager']['performance_metrics']['successful_cycles']}/{stats['heartbeat_manager']['performance_metrics']['total_heartbeats']}")
            
            # Get health status
            health = heartbeat_manager.get_health_status()
            logger.info(f"🏥 Overall Health: {health['overall_health']}")
            logger.info(f"💓 Heartbeat Healthy: {health['heartbeat_healthy']}")
            logger.info(f"💾 Memory Healthy: {health['memory_healthy']}")
            logger.info(f"🌡️  Thermal Healthy: {health['thermal_healthy']}")
            
            # Wait before next cycle (shorter for testing)
            if cycle < 2:  # Don't wait after the last cycle
                logger.info("⏳ Waiting 30 seconds before next cycle...")
                await asyncio.sleep(30)
        
        # Display final statistics
        logger.info(f"\n{'='*60}")
        logger.info("📊 FINAL INTEGRATION STATISTICS")
        logger.info(f"{'='*60}")
        
        final_stats = heartbeat_manager.get_integration_stats()
        
        # Heartbeat Manager Stats
        hb_stats = final_stats['heartbeat_manager']
        logger.info(f"💓 Heartbeat Manager:")
        logger.info(f"   - Total Heartbeats: {hb_stats['heartbeat_count']}")
        logger.info(f"   - Successful Cycles: {hb_stats['performance_metrics']['successful_cycles']}")
        logger.info(f"   - Failed Cycles: {hb_stats['performance_metrics']['failed_cycles']}")
        logger.info(f"   - Average Cycle Time: {hb_stats['performance_metrics']['average_cycle_time']:.3f}s")
        logger.info(f"   - System Uptime: {hb_stats['performance_metrics']['system_uptime']:.1f}s")
        
        # Module Status
        logger.info(f"🔧 Module Status:")
        for module, available in final_stats['modules'].items():
            status = "✅ Available" if available else "❌ Not Available"
            logger.info(f"   - {module}: {status}")
        
        # Strategy Performance
        logger.info(f"🎯 Strategy Performance:")
        logger.info(f"   - Active Strategies: {final_stats['active_strategies']}")
        logger.info(f"   - Strategy Performance Entries: {len(final_stats['strategy_performance'])}")
        
        # History Tracking
        logger.info(f"📚 History Tracking:")
        logger.info(f"   - Thermal History: {final_stats['thermal_history']} entries")
        logger.info(f"   - Memory History: {final_stats['memory_history']} entries")
        
        # Module-specific stats
        if 'thermal_router_stats' in final_stats:
            tr_stats = final_stats['thermal_router_stats']
            logger.info(f"🌡️  Thermal Router Stats:")
            logger.info(f"   - Current Mode: {tr_stats.get('current_mode', 'unknown')}")
            logger.info(f"   - ZPE: {tr_stats.get('zpe', 0.0):.3f}")
            logger.info(f"   - ZBE: {tr_stats.get('zbe', 0.0):.3f}")
        
        if 'autonomic_layer_stats' in final_stats:
            al_stats = final_stats['autonomic_layer_stats']
            logger.info(f"🛡️  Autonomic Layer Stats:")
            logger.info(f"   - Total Validations: {al_stats.get('total_validations', 0)}")
            logger.info(f"   - Blocked Strategies: {al_stats.get('blocked_strategies', 0)}")
            logger.info(f"   - Successful Executions: {al_stats.get('successful_executions', 0)}")
        
        if 'gpu_mapper_stats' in final_stats:
            gpu_stats = final_stats['gpu_mapper_stats']
            logger.info(f"🎮 GPU Logic Mapper Stats:")
            logger.info(f"   - GPU Available: {gpu_stats.get('gpu_available', False)}")
            logger.info(f"   - Mapped Strategies: {gpu_stats.get('mapped_strategies_count', 0)}")
            logger.info(f"   - GPU Memory Usage: {gpu_stats.get('gpu_memory_usage_mb', 0.0):.2f}MB")
            logger.info(f"   - GPU Memory Usage %: {gpu_stats.get('gpu_memory_usage_percent', 0.0):.1f}%")
        
        if 'profit_engine_stats' in final_stats:
            pe_stats = final_stats['profit_engine_stats']
            logger.info(f"💰 Profit Projection Engine Stats:")
            logger.info(f"   - Total Projections: {pe_stats.get('performance_metrics', {}).get('total_projections', 0)}")
            logger.info(f"   - GPU Projections: {pe_stats.get('performance_metrics', {}).get('gpu_projections', 0)}")
            logger.info(f"   - Average Accuracy: {pe_stats.get('performance_metrics', {}).get('average_accuracy', 0.0):.3f}")
        
        # Health Status
        final_health = heartbeat_manager.get_health_status()
        logger.info(f"🏥 Final Health Status:")
        logger.info(f"   - Overall Health: {final_health['overall_health']}")
        logger.info(f"   - Success Rate: {final_health['success_rate']:.3f}")
        logger.info(f"   - Last Heartbeat Age: {final_health['last_heartbeat_age']:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the manager
        logger.info("🛑 Stopping Heartbeat Integration Manager...")
        await heartbeat_manager.stop()
        logger.info("✅ Test completed")


async def test_continuous_heartbeat():
    """Test continuous heartbeat operation."""
    logger.info("🔄 Starting Continuous Heartbeat Test")
    
    # Initialize the heartbeat integration manager
    heartbeat_manager = HeartbeatIntegrationManager()
    
    try:
        # Initialize and start
        if not await heartbeat_manager.initialize():
            logger.error("❌ Failed to initialize")
            return
        
        if not await heartbeat_manager.start():
            logger.error("❌ Failed to start")
            return
        
        # Run continuous heartbeat for a short time
        logger.info("💓 Running continuous heartbeat for 2 minutes...")
        
        # Create a task for continuous heartbeat
        heartbeat_task = asyncio.create_task(heartbeat_manager.run_continuous_heartbeat())
        
        # Monitor for 2 minutes
        start_time = time.time()
        while time.time() - start_time < 120:  # 2 minutes
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Get current stats
            stats = heartbeat_manager.get_integration_stats()
            health = heartbeat_manager.get_health_status()
            
            logger.info(f"📊 Heartbeats: {stats['heartbeat_manager']['heartbeat_count']}, "
                       f"Health: {health['overall_health']}, "
                       f"Success Rate: {health['success_rate']:.3f}")
        
        # Cancel the heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        logger.error(f"❌ Continuous test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await heartbeat_manager.stop()
        logger.info("✅ Continuous test completed")


def test_individual_modules():
    """Test individual modules separately."""
    logger.info("🧪 Testing Individual Modules")
    
    try:
        # Test GPU Logic Mapper
        logger.info("🎮 Testing GPU Logic Mapper...")
        from core.gpu_logic_mapper import GPULogicMapper
        
        gpu_mapper = GPULogicMapper()
        test_hash = "test_strategy_hash_12345"
        
        result = gpu_mapper.map_strategy_to_gpu(test_hash)
        logger.info(f"   - Mapping Status: {result['status']}")
        logger.info(f"   - GPU Memory Usage: {result['gpu_memory_usage']:.2f}MB")
        logger.info(f"   - Matrix Size: {result['matrix_size']}")
        
        gpu_stats = gpu_mapper.get_gpu_stats()
        logger.info(f"   - GPU Available: {gpu_stats['gpu_available']}")
        logger.info(f"   - Mapped Strategies: {gpu_stats['mapped_strategies_count']}")
        
        # Test Profit Projection Engine
        logger.info("💰 Testing Profit Projection Engine...")
        from core.profit_projection_engine import ProfitProjectionEngine
        
        profit_engine = ProfitProjectionEngine()
        
        # Update market conditions
        market_data = {
            "volatility": 0.3,
            "trend": "bullish",
            "volume": 1500.0,
            "sentiment": 0.6
        }
        profit_engine.update_market_conditions(market_data)
        
        # Test profit projection
        strategy_data = {
            "hash": test_hash,
            "tag": "test_strategy",
            "risk_level": 1.0
        }
        
        projection = profit_engine.project_profit(strategy_data)
        logger.info(f"   - Projected Profit: {projection:.3f}%")
        
        # Add some profit data
        profit_engine.add_profit_data("test_strategy", 2.5)
        profit_engine.add_profit_data("test_strategy", 1.8)
        profit_engine.add_profit_data("test_strategy", 3.2)
        
        # Test projection with historical data
        projection_with_history = profit_engine.project_profit(strategy_data)
        logger.info(f"   - Projection with History: {projection_with_history:.3f}%")
        
        engine_stats = profit_engine.get_engine_stats()
        logger.info(f"   - Total Projections: {engine_stats['performance_metrics']['total_projections']}")
        logger.info(f"   - GPU Projections: {engine_stats['performance_metrics']['gpu_projections']}")
        
    except Exception as e:
        logger.error(f"❌ Individual module test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    logger.info("🧪 SCHWABOT HEARTBEAT INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    # Test individual modules first
    test_individual_modules()
    
    print("\n" + "=" * 60)
    
    # Test heartbeat integration
    await test_heartbeat_integration()
    
    print("\n" + "=" * 60)
    
    # Test continuous heartbeat (optional - uncomment to run)
    # await test_continuous_heartbeat()
    
    logger.info("🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main()) 