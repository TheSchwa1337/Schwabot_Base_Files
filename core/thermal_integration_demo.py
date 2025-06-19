"""
Thermal Integration Demo
=======================

Demonstration script showing how to integrate the comprehensive thermal monitoring
and management system with the existing Schwabot architecture. This script provides
examples of:

1. Setting up the thermal system integration
2. Connecting with existing visual controllers
3. Recording trading events with thermal context
4. Using the thermal dashboard
5. API endpoint integration
6. Real-time monitoring and alerts

Usage:
    python thermal_integration_demo.py
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Core thermal system imports
from .thermal_system_integration import (
    ThermalSystemIntegration, 
    ThermalSystemConfig,
    create_thermal_system_integration
)
from .thermal_performance_tracker import TickEventType
from .unified_visual_controller import UnifiedVisualController
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
from .gpu_metrics import GPUMetrics
from .flask_gateway import Flask, jsonify

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThermalIntegrationDemo:
    """
    Demonstration of thermal system integration with Schwabot
    """
    
    def __init__(self):
        """Initialize the demo system"""
        # Create Flask app for API endpoints
        self.flask_app = Flask(__name__)
        
        # Create core components
        self.profit_coprocessor = ProfitTrajectoryCoprocessor(window_size=1000)
        self.visual_controller = self._create_mock_visual_controller()
        self.gpu_metrics = self._create_mock_gpu_metrics()
        
        # Configure thermal system
        self.thermal_config = ThermalSystemConfig(
            monitoring_interval=2.0,  # Update every 2 seconds for demo
            visual_update_interval=1.0,  # Visual updates every second
            enable_api_endpoints=True,
            enable_visual_integration=True,
            enable_hover_portals=True,
            export_data_enabled=True,
            thermal_thresholds={
                'cpu_warning': 75.0,
                'cpu_critical': 85.0,
                'gpu_warning': 80.0,
                'gpu_critical': 90.0
            }
        )
        
        # Create thermal system integration
        self.thermal_system = create_thermal_system_integration(
            visual_controller=self.visual_controller,
            profit_coprocessor=self.profit_coprocessor,
            gpu_metrics=self.gpu_metrics,
            config=self.thermal_config,
            flask_app=self.flask_app
        )
        
        # Demo state
        self.demo_running = False
        self.trade_counter = 0
        self.demo_scenarios = [
            self._normal_operation_scenario,
            self._high_temperature_scenario,
            self._burst_processing_scenario,
            self._thermal_emergency_scenario
        ]
        self.current_scenario = 0
        
        logger.info("ThermalIntegrationDemo initialized")
    
    def _create_mock_visual_controller(self) -> UnifiedVisualController:
        """Create a mock visual controller for demonstration"""
        class MockVisualController:
            def __init__(self):
                self.thermal_widgets = {}
                self.thermal_warnings = []
            
            def update_thermal_widgets(self, widget_data: Dict[str, Any]) -> None:
                """Update thermal widgets with new data"""
                self.thermal_widgets.update(widget_data)
                logger.info(f"Visual controller updated with thermal data: {len(widget_data)} items")
            
            def show_thermal_warning(self, thermal_state: Dict[str, Any]) -> None:
                """Show thermal warning in UI"""
                warning = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'thermal_zone': thermal_state.get('thermal_zone', 'unknown'),
                    'cpu_temp': thermal_state.get('cpu_temperature', 0),
                    'gpu_temp': thermal_state.get('gpu_temperature', 0)
                }
                self.thermal_warnings.append(warning)
                logger.warning(f"THERMAL WARNING: {warning}")
        
        return MockVisualController()
    
    def _create_mock_gpu_metrics(self) -> GPUMetrics:
        """Create mock GPU metrics for demonstration"""
        class MockGPUMetrics:
            def __init__(self):
                self.base_utilization = 50.0
                self.utilization_trend = 0.0
            
            def get_gpu_utilization(self) -> float:
                """Get simulated GPU utilization"""
                # Add some realistic variation
                self.utilization_trend += random.uniform(-2, 2)
                self.utilization_trend = max(-10, min(10, self.utilization_trend))
                
                utilization = self.base_utilization + self.utilization_trend + random.uniform(-5, 5)
                return max(0, min(100, utilization))
            
            def get_gpu_temperature(self) -> float:
                """Get simulated GPU temperature"""
                base_temp = 65.0
                load_factor = self.get_gpu_utilization() / 100.0
                return base_temp + (load_factor * 20) + random.uniform(-3, 3)
        
        return MockGPUMetrics()
    
    async def run_demo(self) -> None:
        """Run the complete thermal integration demonstration"""
        logger.info("Starting Thermal Integration Demo")
        
        try:
            # Start the thermal system
            success = await self.thermal_system.start_system()
            if not success:
                logger.error("Failed to start thermal system")
                return
            
            self.demo_running = True
            
            # Run demonstration scenarios
            await self._run_demo_scenarios()
            
            # Show final statistics
            self._show_final_statistics()
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Error in demo: {e}")
        finally:
            # Stop the thermal system
            await self.thermal_system.stop_system()
            self.demo_running = False
            logger.info("Thermal Integration Demo completed")
    
    async def _run_demo_scenarios(self) -> None:
        """Run different demonstration scenarios"""
        scenario_duration = 30  # 30 seconds per scenario
        
        for i, scenario in enumerate(self.demo_scenarios):
            logger.info(f"Running scenario {i+1}/{len(self.demo_scenarios)}: {scenario.__name__}")
            
            # Run scenario
            scenario_task = asyncio.create_task(scenario())
            trading_task = asyncio.create_task(self._simulate_trading_activity())
            
            # Run for specified duration
            await asyncio.sleep(scenario_duration)
            
            # Cancel tasks
            scenario_task.cancel()
            trading_task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(scenario_task, trading_task, return_exceptions=True)
            
            # Show scenario statistics
            self._show_scenario_statistics(i+1)
            
            if i < len(self.demo_scenarios) - 1:
                logger.info("Waiting 5 seconds before next scenario...")
                await asyncio.sleep(5)
    
    async def _normal_operation_scenario(self) -> None:
        """Simulate normal operation with stable temperatures"""
        while self.demo_running:
            # Add some profit data
            profit = 1000 + random.uniform(-50, 50)
            self.profit_coprocessor.update(profit)
            
            # Simulate normal system events
            if random.random() < 0.1:  # 10% chance
                self.thermal_system.performance_tracker.record_tick_event(
                    TickEventType.THERMAL_UPDATE,
                    {'scenario': 'normal_operation', 'temperature_stable': True}
                )
            
            await asyncio.sleep(2)
    
    async def _high_temperature_scenario(self) -> None:
        """Simulate high temperature conditions"""
        logger.info("Simulating high temperature conditions...")
        
        while self.demo_running:
            # Simulate higher temperatures by adjusting GPU metrics
            if hasattr(self.gpu_metrics, 'base_utilization'):
                self.gpu_metrics.base_utilization = min(90, self.gpu_metrics.base_utilization + 1)
            
            # Record thermal events
            self.thermal_system.performance_tracker.record_tick_event(
                TickEventType.THERMAL_UPDATE,
                {'scenario': 'high_temperature', 'temperature_rising': True}
            )
            
            await asyncio.sleep(3)
    
    async def _burst_processing_scenario(self) -> None:
        """Simulate burst processing with thermal management"""
        logger.info("Simulating burst processing scenario...")
        
        while self.demo_running:
            # Simulate burst processing
            if self.thermal_system.thermal_manager._can_burst():
                logger.info("Starting burst processing...")
                
                success = self.thermal_system.thermal_manager.start_burst()
                if success:
                    self.thermal_system.performance_tracker.record_tick_event(
                        TickEventType.BURST_START,
                        {'scenario': 'burst_processing', 'duration_planned': 10}
                    )
                    
                    # Simulate burst work
                    await asyncio.sleep(10)
                    
                    # End burst
                    self.thermal_system.thermal_manager.end_burst(10.0)
                    self.thermal_system.performance_tracker.record_tick_event(
                        TickEventType.BURST_END,
                        {'scenario': 'burst_processing', 'duration_actual': 10}
                    )
                    
                    logger.info("Burst processing completed")
            
            await asyncio.sleep(15)
    
    async def _thermal_emergency_scenario(self) -> None:
        """Simulate thermal emergency conditions"""
        logger.info("Simulating thermal emergency scenario...")
        
        while self.demo_running:
            # Force high temperatures
            if hasattr(self.gpu_metrics, 'base_utilization'):
                self.gpu_metrics.base_utilization = 95
            
            # Record emergency events
            self.thermal_system.performance_tracker.record_tick_event(
                TickEventType.ERROR_EVENT,
                {
                    'scenario': 'thermal_emergency',
                    'type': 'critical_temperature',
                    'action_required': True
                }
            )
            
            await asyncio.sleep(5)
    
    async def _simulate_trading_activity(self) -> None:
        """Simulate trading activity with thermal context"""
        while self.demo_running:
            # Simulate trading decisions
            actions = ['buy', 'sell', 'hold']
            action = random.choice(actions)
            amount = random.uniform(100, 1000)
            confidence = random.uniform(0.6, 0.95)
            
            # Record trading event with thermal context
            self.thermal_system.record_trading_event(
                action=action,
                amount=amount,
                confidence=confidence,
                metadata={
                    'demo_trade': True,
                    'trade_id': self.trade_counter,
                    'scenario': self.current_scenario
                }
            )
            
            self.trade_counter += 1
            
            # Wait random interval
            await asyncio.sleep(random.uniform(3, 8))
    
    def _show_scenario_statistics(self, scenario_num: int) -> None:
        """Show statistics for completed scenario"""
        stats = self.thermal_system.get_system_statistics()
        
        logger.info(f"=== Scenario {scenario_num} Statistics ===")
        logger.info(f"Total Events: {stats.get('total_events_processed', 0)}")
        logger.info(f"Thermal Events: {stats.get('thermal_events', 0)}")
        logger.info(f"Trade Decisions: {stats.get('trade_decisions', 0)}")
        logger.info(f"Burst Events: {stats.get('burst_events', 0)}")
        logger.info(f"System Health: {stats.get('system_health_average', 1.0):.1%}")
        
        # Show thermal recommendations
        recommendations = self.thermal_system.get_thermal_recommendations()
        if recommendations:
            logger.info("Current Recommendations:")
            for rec in recommendations:
                logger.info(f"  - {rec}")
        
        logger.info("=" * 40)
    
    def _show_final_statistics(self) -> None:
        """Show final demonstration statistics"""
        stats = self.thermal_system.get_system_statistics()
        
        logger.info("=== FINAL DEMO STATISTICS ===")
        logger.info(f"Total Runtime: {stats.get('uptime_seconds', 0):.1f} seconds")
        logger.info(f"Total Events Processed: {stats.get('total_events_processed', 0)}")
        logger.info(f"Thermal Warnings: {stats.get('thermal_warnings', 0)}")
        logger.info(f"Thermal Criticals: {stats.get('thermal_criticals', 0)}")
        logger.info(f"Average System Health: {stats.get('system_health_average', 1.0):.1%}")
        logger.info(f"Total Trades Simulated: {self.trade_counter}")
        
        # Show thermal manager statistics
        if 'thermal_manager' in stats:
            thermal_stats = stats['thermal_manager']
            logger.info(f"Burst Events: {thermal_stats.get('burst_statistics', {}).get('total_bursts', 0)}")
            logger.info(f"Total Burst Time: {thermal_stats.get('burst_statistics', {}).get('total_burst_time', 0):.1f}s")
        
        # Show visual controller statistics
        if hasattr(self.visual_controller, 'thermal_warnings'):
            logger.info(f"Visual Warnings Displayed: {len(self.visual_controller.thermal_warnings)}")
        
        logger.info("=" * 50)
    
    def get_dashboard_url(self) -> str:
        """Get URL for thermal dashboard"""
        return "http://localhost:5000/api/thermal/dashboard"
    
    def export_demo_data(self, filepath: str = "thermal_demo_export.json") -> bool:
        """Export demonstration data"""
        logger.info(f"Exporting demo data to {filepath}")
        return self.thermal_system.performance_tracker.export_data(filepath)
    
    def start_flask_server(self, host: str = "localhost", port: int = 5000) -> None:
        """Start Flask server for API endpoints"""
        logger.info(f"Starting Flask server at http://{host}:{port}")
        
        # Add a simple index route
        @self.flask_app.route('/')
        def index():
            return f"""
            <html>
            <head><title>Thermal Integration Demo</title></head>
            <body>
                <h1>Schwabot Thermal Integration Demo</h1>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><a href="/api/thermal/status">Thermal Status</a></li>
                    <li><a href="/api/thermal/dashboard">Thermal Dashboard</a></li>
                    <li><a href="/api/thermal/visualization-data">Visualization Data</a></li>
                </ul>
                <h2>System Information:</h2>
                <p>Demo Running: {self.demo_running}</p>
                <p>Total Trades: {self.trade_counter}</p>
                <p>System Healthy: {self.thermal_system.is_system_healthy()}</p>
            </body>
            </html>
            """
        
        self.flask_app.run(host=host, port=port, debug=False)

async def main():
    """Main demonstration function"""
    demo = ThermalIntegrationDemo()
    
    logger.info("Thermal Integration Demo Starting...")
    logger.info("This demo will show:")
    logger.info("1. Normal operation with thermal monitoring")
    logger.info("2. High temperature scenario")
    logger.info("3. Burst processing with thermal management")
    logger.info("4. Thermal emergency handling")
    logger.info("")
    logger.info("Press Ctrl+C to stop the demo at any time")
    logger.info("=" * 60)
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    
    # Export demo data
    demo.export_demo_data()
    
    logger.info("Demo completed. Check thermal_demo_export.json for detailed data.")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 