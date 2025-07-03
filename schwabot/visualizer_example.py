#!/usr/bin/env python3
"""
Schwabot Visualization Example
==============================

This example demonstrates how to use the Schwabot visualization system
to monitor and visualize trading operations, mathematical calculations,
and system performance in real-time.
"""

import asyncio
import time
import random
import logging
from typing import Dict, Any

from .integration_hub import create_schwabot_hub
from .visualizer import (
    SchwabotVisualizer,
    DataAggregator,
    PerformanceMonitor,
    TradingVisualizer,
    MathVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class VisualizationExample:
    """
    Example class demonstrating Schwabot visualization capabilities.
    """
    
    def __init__(self):
        """Initialize the visualization example"""
        self.logger = logging.getLogger('VisualizationExample')
        self.hub = None
        
    async def setup_visualization(self):
        """Set up the visualization system"""
        self.logger.info("Setting up Schwabot visualization system...")
        
        # Create the integration hub with visualization enabled
        self.hub = await create_schwabot_hub(
            initial_capital=100000.0,
            debug=True,
            enable_visualization=True
        )
        
        # Register visualization callbacks
        if self.hub.visualizer:
            self.hub.visualizer.register_visualization_callback(self.on_visualization_event)
        
        if self.hub.trading_visualizer:
            self.hub.trading_visualizer.register_trading_callback(self.on_trading_event)
        
        if self.hub.math_visualizer:
            self.hub.math_visualizer.register_math_callback(self.on_math_event)
        
        if self.hub.performance_monitor:
            self.hub.performance_monitor.register_metric_callback(self.on_performance_event)
        
        self.logger.info("Visualization system setup complete")
    
    def on_visualization_event(self, event_type: str, data: Dict[str, Any]):
        """Handle visualization events"""
        if event_type == 'periodic_update':
            self.logger.info(f"Visualization update: {len(data.get('recent_events', []))} events")
        else:
            self.logger.info(f"Visualization event: {event_type}")
    
    def on_trading_event(self, event_type: str, data: Dict[str, Any]):
        """Handle trading events"""
        if event_type == 'trade_execution':
            self.logger.info(f"Trade executed: {data.get('symbol')} {data.get('side')} {data.get('amount')}")
        elif event_type == 'periodic_update':
            summary = data.get('trading_summary', {})
            self.logger.info(f"Trading summary: {summary.get('total_trades', 0)} total trades")
    
    def on_math_event(self, event_type: str, data: Dict[str, Any]):
        """Handle mathematical events"""
        if event_type == 'calculation_completed':
            self.logger.info(f"Math calculation: {data.get('operation')} completed in {data.get('duration', 0):.3f}s")
        elif event_type == 'tensor_operation':
            self.logger.info(f"Tensor operation: {data.get('operation')} with {data.get('flops', 0)} FLOPs")
    
    def on_performance_event(self, metrics: Dict[str, Any]):
        """Handle performance events"""
        system_metrics = metrics.get('system', {})
        self.logger.info(f"Performance: CPU {system_metrics.get('cpu_percent', 0):.1f}%, "
                        f"Memory {system_metrics.get('memory_percent', 0):.1f}%")
    
    async def simulate_trading_activity(self, duration: int = 60):
        """Simulate trading activity for demonstration"""
        self.logger.info(f"Starting trading simulation for {duration} seconds...")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT']
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Simulate trade execution
            symbol = random.choice(symbols)
            side = random.choice(['buy', 'sell'])
            amount = random.uniform(0.1, 2.0)
            price = random.uniform(30000, 50000) if 'BTC' in symbol else random.uniform(2000, 4000)
            
            if self.hub and self.hub.trading_visualizer:
                self.hub.trading_visualizer.add_trade_execution(
                    symbol, side, amount, price, f"order_{int(time.time())}"
                )
            
            # Simulate market data
            if self.hub and self.hub.trading_visualizer:
                self.hub.trading_visualizer.add_market_data(symbol, {
                    'price': price,
                    'volume': random.uniform(1000, 10000),
                    'change_24h': random.uniform(-5, 5)
                })
            
            # Simulate order book data
            if self.hub and self.hub.trading_visualizer:
                bids = [{'price': price * 0.999, 'amount': random.uniform(1, 10)} for _ in range(5)]
                asks = [{'price': price * 1.001, 'amount': random.uniform(1, 10)} for _ in range(5)]
                self.hub.trading_visualizer.add_order_book_data(symbol, bids, asks)
            
            await asyncio.sleep(random.uniform(1, 3))
        
        self.logger.info("Trading simulation completed")
    
    async def simulate_mathematical_calculations(self, duration: int = 60):
        """Simulate mathematical calculations for demonstration"""
        self.logger.info(f"Starting mathematical simulation for {duration} seconds...")
        
        operations = ['matrix_multiply', 'tensor_convolution', 'eigenvalue_decomposition', 'fourier_transform']
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Simulate mathematical calculation
            operation = random.choice(operations)
            inputs = {'size': random.randint(100, 1000), 'iterations': random.randint(1, 10)}
            result = random.uniform(0, 1000)
            duration_calc = random.uniform(0.001, 0.1)
            
            if self.hub and self.hub.math_visualizer:
                self.hub.math_visualizer.add_calculation(
                    operation, inputs, result, duration_calc, 'O(n^2)'
                )
            
            # Simulate tensor operation
            if self.hub and self.hub.math_visualizer:
                tensor_shape = [random.randint(10, 100) for _ in range(random.randint(2, 4))]
                result_shape = [random.randint(10, 100) for _ in range(random.randint(2, 4))]
                self.hub.math_visualizer.add_tensor_operation(
                    operation, tensor_shape, result_shape, duration_calc, random.uniform(1000000, 10000000)
                )
            
            # Simulate algorithm performance
            if self.hub and self.hub.math_visualizer:
                self.hub.math_visualizer.add_algorithm_performance(
                    f"algorithm_{random.randint(1, 5)}",
                    random.randint(1000, 10000),
                    duration_calc,
                    random.uniform(1000000, 10000000),
                    random.randint(1, 5)
                )
            
            await asyncio.sleep(random.uniform(0.5, 2))
        
        self.logger.info("Mathematical simulation completed")
    
    async def run_demonstration(self, duration: int = 120):
        """Run the complete demonstration"""
        try:
            # Set up visualization
            await self.setup_visualization()
            
            # Run simulations in parallel
            await asyncio.gather(
                self.simulate_trading_activity(duration),
                self.simulate_mathematical_calculations(duration)
            )
            
            # Get final status
            status = self.hub.get_system_status()
            self.logger.info("Final system status:")
            self.logger.info(f"  Capital: ${status['current_capital']:,.2f}")
            self.logger.info(f"  Tensors collected: {status['collected_tensors']}")
            
            if 'visualization' in status:
                viz_status = status['visualization']
                self.logger.info("  Visualization status:")
                self.logger.info(f"    Total events: {viz_status['visualizer'].get('performance_metrics', {}).get('total_events', 0)}")
                self.logger.info(f"    Trading summary: {viz_status['trading'].get('trading_summary', {}).get('total_trades', 0)} trades")
                self.logger.info(f"    Math calculations: {viz_status['math'].get('calculations', {}).get('total_calculations', 0)}")
            
        except Exception as e:
            self.logger.error(f"Demonstration error: {e}")
        finally:
            # Cleanup
            if self.hub:
                await self.hub.shutdown()

async def main():
    """Main entry point for the visualization example"""
    example = VisualizationExample()
    await example.run_demonstration(duration=60)  # Run for 60 seconds

if __name__ == '__main__':
    asyncio.run(main()) 