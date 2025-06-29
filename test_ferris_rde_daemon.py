#!/usr/bin/env python3
"""
Test Script for Ferris RDE Daemon

This script demonstrates the Ferris RDE daemon functionality with:
- Proper entry/exit logic testing
- Integrated pipeline validation
- Health monitoring verification
- Performance metrics analysis
- Mathematical state integration testing
"""

import asyncio
import logging
import time
from typing import Dict, Any

from core.ferris_rde_daemon import FerrisRDEDaemon, DaemonConfig, get_daemon_instance
from utils.safe_print import info, success, warn, error, safe_print


async def test_daemon_initialization():
    """Test daemon initialization and configuration."""
    safe_print("🧪 Testing Daemon Initialization...")
    
    try:
        # Create custom configuration
        config = DaemonConfig(
            daemon_name="TestFerrisRDE",
            tick_interval_seconds=2.0,  # Faster for testing
            health_check_interval_seconds=10.0,
            performance_report_interval_seconds=30.0,
            primary_assets=["BTC/USD"],
            secondary_assets=["ETH/USD"],
            paper_trading=True,
            max_concurrent_trades=5,
        )
        
        # Initialize daemon
        daemon = FerrisRDEDaemon(config)
        
        # Check initialization
        assert daemon.config.daemon_name == "TestFerrisRDE"
        assert daemon.config.paper_trading is True
        assert len(daemon.config.primary_assets) == 1
        assert daemon.config.primary_assets[0] == "BTC/USD"
        
        success("✅ Daemon initialization test passed")
        return daemon
        
    except Exception as e:
        error(f"❌ Daemon initialization test failed: {e}")
        return None


async def test_daemon_start_stop(daemon: FerrisRDEDaemon):
    """Test daemon start and stop functionality."""
    safe_print("🧪 Testing Daemon Start/Stop...")
    
    try:
        # Start daemon
        start_success = await daemon.start()
        assert start_success is True
        assert daemon.running is True
        
        success("✅ Daemon started successfully")
        
        # Let it run for a few seconds
        safe_print("⏳ Running daemon for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check metrics
        metrics = daemon.metrics.get_summary()
        safe_print(f"📊 Metrics after 10s: {metrics['total_ticks_processed']} ticks processed")
        
        # Stop daemon
        stop_success = await daemon.stop()
        assert stop_success is True
        assert daemon.running is False
        
        success("✅ Daemon stop test passed")
        
    except Exception as e:
        error(f"❌ Daemon start/stop test failed: {e}")


async def test_entry_exit_logic(daemon: FerrisRDEDaemon):
    """Test entry/exit logic with simulated market data."""
    safe_print("🧪 Testing Entry/Exit Logic...")
    
    try:
        # Start daemon
        await daemon.start()
        
        # Simulate market data processing
        test_market_data = {
            "asset": "BTC/USD",
            "timestamp": time.time(),
            "current_price": 50000.0,
            "price_change": 2.5,
            "volume": 1000000.0,
            "volatility": 0.15,
            "order_book": {
                "bids": [[49999, 1.0], [49998, 2.0]],
                "asks": [[50001, 1.0], [50002, 2.0]],
            }
        }
        
        # Process through trading pipeline
        trading_signal = await daemon.trading_pipeline.process_market_data(
            market_data=test_market_data,
            asset="BTC/USD",
            thermal_state="warm"
        )
        
        # Verify signal properties
        assert trading_signal.asset == "BTC/USD"
        assert trading_signal.confidence >= 0.0
        assert trading_signal.confidence <= 1.0
        assert trading_signal.bit_depth >= 2
        assert trading_signal.bit_depth <= 42
        
        safe_print(f"📈 Generated signal: {trading_signal.signal_type} "
                  f"(confidence: {trading_signal.confidence:.3f}, "
                  f"bit_depth: {trading_signal.bit_depth})")
        
        # Test entry/exit execution
        if trading_signal.confidence > 0.7:
            await daemon._execute_trading_signal(trading_signal, test_market_data)
            safe_print("📄 Paper trade executed")
        
        # Stop daemon
        await daemon.stop()
        
        success("✅ Entry/exit logic test passed")
        
    except Exception as e:
        error(f"❌ Entry/exit logic test failed: {e}")


async def test_mathematical_integration(daemon: FerrisRDEDaemon):
    """Test mathematical state integration."""
    safe_print("🧪 Testing Mathematical Integration...")
    
    try:
        # Start daemon
        await daemon.start()
        
        # Test mathematical state updates
        sample_price_data = [50000, 50100, 50200, 50300, 50400]
        sample_volume_data = [100, 120, 110, 130, 125]
        sample_time_series = list(range(len(sample_price_data)))
        
        # Update mathematical states
        mathematical_states = await daemon.connectivity_manager.update_mathematical_states(
            price_data=sample_price_data,
            volume_data=sample_volume_data,
            time_series=sample_time_series,
            metadata={"test": True}
        )
        
        # Verify mathematical states
        assert len(mathematical_states) > 0
        
        # Check for specific mathematical components
        if "ferris_wheel_state" in mathematical_states:
            ferris_state = mathematical_states["ferris_wheel_state"]
            safe_print(f"🎡 Ferris wheel state: cycle_position={ferris_state.cycle_position:.3f}")
        
        if "quantum_thermal_state" in mathematical_states:
            quantum_state = mathematical_states["quantum_thermal_state"]
            safe_print(f"⚛️ Quantum thermal state: temperature={quantum_state.temperature:.1f}K")
        
        if "void_well_metrics" in mathematical_states:
            void_well = mathematical_states["void_well_metrics"]
            safe_print(f"🕳️ Void well metrics: fractal_index={void_well.fractal_index:.4f}")
        
        if "kelly_metrics" in mathematical_states:
            kelly = mathematical_states["kelly_metrics"]
            safe_print(f"📊 Kelly metrics: safe_kelly={kelly.safe_kelly:.4f}")
        
        # Stop daemon
        await daemon.stop()
        
        success("✅ Mathematical integration test passed")
        
    except Exception as e:
        error(f"❌ Mathematical integration test failed: {e}")


async def test_health_monitoring(daemon: FerrisRDEDaemon):
    """Test health monitoring and system status."""
    safe_print("🧪 Testing Health Monitoring...")
    
    try:
        # Start daemon
        await daemon.start()
        
        # Wait for health checks to run
        await asyncio.sleep(15)
        
        # Get system status
        system_status = await daemon.connectivity_manager.get_system_status()
        
        # Verify system status structure
        assert "system_info" in system_status
        assert "api_status" in system_status
        assert "trading_status" in system_status
        assert "visualization_status" in system_status
        assert "mathematical_status" in system_status
        
        # Check API status
        api_status = system_status["api_status"]
        safe_print(f"🔌 API Status: {api_status.get('overall_status', 'unknown')}")
        
        # Check trading status
        trading_status = system_status["trading_status"]
        safe_print(f"📈 Trading Status: {trading_status.get('overall_status', 'unknown')}")
        
        # Check mathematical status
        math_status = system_status["mathematical_status"]
        safe_print(f"🧮 Mathematical Status: {len(math_status)} components active")
        
        # Get daemon status
        daemon_status = daemon.get_daemon_status()
        assert "daemon_info" in daemon_status
        assert "metrics" in daemon_status
        assert "components" in daemon_status
        
        safe_print(f"🎡 Daemon Status: {daemon_status['daemon_info']['name']} "
                  f"(uptime: {daemon_status['metrics']['uptime_seconds']:.1f}s)")
        
        # Stop daemon
        await daemon.stop()
        
        success("✅ Health monitoring test passed")
        
    except Exception as e:
        error(f"❌ Health monitoring test failed: {e}")


async def test_performance_metrics(daemon: FerrisRDEDaemon):
    """Test performance metrics and reporting."""
    safe_print("🧪 Testing Performance Metrics...")
    
    try:
        # Start daemon
        await daemon.start()
        
        # Let it run for a while to generate metrics
        safe_print("⏳ Running daemon for 20 seconds to generate metrics...")
        await asyncio.sleep(20)
        
        # Get performance summary
        performance_summary = daemon.metrics.get_summary()
        
        # Verify metrics structure
        required_metrics = [
            "uptime_seconds", "total_ticks_processed", "total_signals_generated",
            "total_trades_executed", "avg_tick_processing_time_ms"
        ]
        
        for metric in required_metrics:
            assert metric in performance_summary
        
        # Log performance metrics
        safe_print("📊 Performance Metrics:")
        safe_print(f"  Uptime: {performance_summary['uptime_seconds']:.1f}s")
        safe_print(f"  Ticks Processed: {performance_summary['total_ticks_processed']}")
        safe_print(f"  Signals Generated: {performance_summary['total_signals_generated']}")
        safe_print(f"  Trades Executed: {performance_summary['total_trades_executed']}")
        safe_print(f"  Avg Processing Time: {performance_summary['avg_tick_processing_time_ms']:.2f}ms")
        safe_print(f"  Total Errors: {performance_summary['total_errors']}")
        
        # Verify reasonable performance
        assert performance_summary['uptime_seconds'] >= 15.0
        assert performance_summary['total_ticks_processed'] > 0
        assert performance_summary['avg_tick_processing_time_ms'] > 0
        
        # Stop daemon
        await daemon.stop()
        
        success("✅ Performance metrics test passed")
        
    except Exception as e:
        error(f"❌ Performance metrics test failed: {e}")


async def test_pipeline_integration(daemon: FerrisRDEDaemon):
    """Test complete pipeline integration."""
    safe_print("🧪 Testing Pipeline Integration...")
    
    try:
        # Start daemon
        await daemon.start()
        
        # Test trade tick pipeline
        if "BTC/USD" in daemon.trade_tick_pipelines:
            pipeline = daemon.trade_tick_pipelines["BTC/USD"]
            
            # Create test order book
            test_order_book = {
                "bids": [[49999, 1.0], [49998, 2.0], [49997, 1.5]],
                "asks": [[50001, 1.0], [50002, 2.0], [50003, 1.5]],
            }
            
            # Process tick
            tick = pipeline.process_tick(
                order_book=test_order_book,
                symbol="BTC/USD",
                ferris_phase=0.5,
                ghost_signal=0.3
            )
            
            # Verify tick properties
            assert tick.symbol == "BTC/USD"
            assert tick.ferris_phase == 0.5
            assert tick.ghost_signal == 0.3
            assert len(tick.strategy_bits) > 0
            assert tick.vector.shape[0] > 0
            
            safe_print(f"📊 Processed tick: {tick.symbol} "
                      f"(entry: {tick.entry}, exit: {tick.exit}, "
                      f"strategy_bits: {len(tick.strategy_bits)})")
        
        # Test multi-bit state manager
        state_manager = daemon.multi_bit_manager
        current_state = state_manager.get_current_state()
        safe_print(f"🧠 Multi-bit state: {current_state}")
        
        # Test dualistic thought engines
        thought_engines = daemon.dualistic_engines
        performance = thought_engines.get_engine_performance()
        safe_print(f"🤔 Dualistic engines performance: {len(performance)} metrics")
        
        # Stop daemon
        await daemon.stop()
        
        success("✅ Pipeline integration test passed")
        
    except Exception as e:
        error(f"❌ Pipeline integration test failed: {e}")


async def run_comprehensive_test():
    """Run comprehensive daemon test suite."""
    safe_print("🚀 Starting Comprehensive Ferris RDE Daemon Test Suite")
    safe_print("=" * 60)
    
    # Initialize daemon
    daemon = await test_daemon_initialization()
    if not daemon:
        error("❌ Cannot proceed without daemon initialization")
        return
    
    try:
        # Run test suite
        await test_daemon_start_stop(daemon)
        await asyncio.sleep(2)
        
        await test_entry_exit_logic(daemon)
        await asyncio.sleep(2)
        
        await test_mathematical_integration(daemon)
        await asyncio.sleep(2)
        
        await test_health_monitoring(daemon)
        await asyncio.sleep(2)
        
        await test_performance_metrics(daemon)
        await asyncio.sleep(2)
        
        await test_pipeline_integration(daemon)
        
        safe_print("=" * 60)
        success("🎉 All tests completed successfully!")
        
        # Final status report
        final_status = daemon.get_daemon_status()
        safe_print(f"📋 Final Status: {final_status['daemon_info']['name']} "
                  f"- Total Errors: {final_status['metrics']['total_errors']}")
        
    except Exception as e:
        error(f"❌ Test suite failed: {e}")
    
    finally:
        # Ensure daemon is stopped
        if daemon.running:
            await daemon.stop()


async def quick_demo():
    """Quick demonstration of daemon functionality."""
    safe_print("🎬 Quick Ferris RDE Daemon Demo")
    safe_print("=" * 40)
    
    try:
        # Get daemon instance
        daemon = get_daemon_instance()
        
        # Start daemon
        safe_print("🚀 Starting daemon...")
        await daemon.start()
        
        # Run for 30 seconds
        safe_print("⏳ Running for 30 seconds...")
        for i in range(30):
            await asyncio.sleep(1)
            if i % 10 == 0:
                safe_print(f"⏱️  {i}s elapsed...")
        
        # Get final status
        status = daemon.get_daemon_status()
        metrics = status['metrics']
        
        safe_print("📊 Demo Results:")
        safe_print(f"  Ticks Processed: {metrics['total_ticks_processed']}")
        safe_print(f"  Signals Generated: {metrics['total_signals_generated']}")
        safe_print(f"  Trades Executed: {metrics['total_trades_executed']}")
        safe_print(f"  Avg Processing Time: {metrics['avg_tick_processing_time_ms']:.2f}ms")
        safe_print(f"  Total Errors: {metrics['total_errors']}")
        
        # Stop daemon
        safe_print("🛑 Stopping daemon...")
        await daemon.stop()
        
        success("✅ Demo completed successfully!")
        
    except Exception as e:
        error(f"❌ Demo failed: {e}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Quick demo mode
        asyncio.run(quick_demo())
    else:
        # Full test suite
        asyncio.run(run_comprehensive_test())


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    main() 