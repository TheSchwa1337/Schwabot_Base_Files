"""
Complete Integrated Anti-Pole System Demo
=========================================

Comprehensive demonstration of the fully integrated Schwabot system:
- Hash Affinity Vault with correlation tracking
- Advanced Test Harness with market simulation  
- Strategy Execution Mapper with trade generation
- System Clock Sequencer with task coordination
- Thermal management and GPU/CPU optimization
- Real-time dashboard integration

This demo showcases the complete "mathematical skeleton → production infrastructure → live trading intelligence" pipeline.

Usage:
    python demo_complete_integrated_system.py [--duration MINUTES] [--live-mode]
"""

import asyncio
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Core system components
from core.hash_affinity_vault import HashAffinityVault
from core.advanced_test_harness import AdvancedTestHarness
from core.strategy_execution_mapper import StrategyExecutionMapper, StrategyType
from core.system_clock_sequencer import SystemClockSequencer, ScheduleFrequency

# Supporting components (with fallbacks)
try:
    from core.thermal_zone_manager import ThermalZoneManager
    from core.vault_router import VaultRouter
    from core.profit_navigator import ProfitNavigator
    from core.entropy_bridge import EntropyBridge
    from core.dashboard_integration import DashboardIntegration
    FULL_COMPONENTS_AVAILABLE = True
except ImportError:
    FULL_COMPONENTS_AVAILABLE = False

class MockComponent:
    """Mock component for missing dependencies"""
    async def get_thermal_state(self):
        return {'gpu_utilization': 0.3, 'cpu_utilization': 0.2, 'gpu_temperature': 65, 'cpu_temperature': 55}
    
    async def route_strategy(self, params):
        return {'status': 'routed', 'params': params}
    
    async def calculate_position_size(self, volume_fraction, confidence):
        return volume_fraction * 10000.0  # Mock $10k position
    
    async def get_entropy_statistics(self):
        return {'total_entropy_events': 42, 'avg_entropy': 0.65}
    
    async def broadcast_update(self, data):
        logging.getLogger(__name__).debug(f"Dashboard update: {len(str(data))} bytes")

class CompleteIntegratedSystemDemo:
    """
    Complete demonstration of the integrated Anti-Pole trading system
    """
    
    def __init__(self, duration_minutes: int = 10, live_mode: bool = False):
        """
        Initialize the complete integrated system
        
        Args:
            duration_minutes: How long to run the demonstration
            live_mode: Whether to simulate live market conditions
        """
        self.duration_minutes = duration_minutes
        self.live_mode = live_mode
        
        # Initialize logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.vault = None
        self.test_harness = None
        self.strategy_mapper = None
        self.clock_sequencer = None
        
        # Supporting components (with mocks if needed)
        self.thermal_manager = None
        self.vault_router = None
        self.profit_navigator = None
        self.entropy_bridge = None
        self.dashboard = None
        
        # Demo state
        self.demo_stats = {
            'start_time': None,
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'total_vault_ticks': 0,
            'system_health_history': [],
            'strategy_performance': {},
            'thermal_events': 0
        }
        
        # Market simulation state for live mode
        self.current_btc_price = 45000.0
        self.price_history = []
        
    def setup_logging(self):
        """Setup comprehensive logging for the demo"""
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"integrated_system_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        
        # Reduce noise from some modules
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    async def initialize_system(self):
        """Initialize all system components"""
        self.logger.info("Starting Complete Integrated Anti-Pole System")
        
        # Initialize Hash Affinity Vault
        self.vault = HashAffinityVault(max_history=5000, correlation_window=200)
        self.logger.info("Hash Affinity Vault initialized")
        
        # Initialize supporting components (real or mock)
        components_loaded = False
        if FULL_COMPONENTS_AVAILABLE:
            try:
                self.thermal_manager = ThermalZoneManager()
                self.vault_router = VaultRouter()
                self.profit_navigator = ProfitNavigator()
                self.entropy_bridge = EntropyBridge()
                self.dashboard = DashboardIntegration()
                self.logger.info("Full component suite loaded")
                components_loaded = True
            except Exception as e:
                self.logger.warning(f"Failed to load full components: {e}")
                components_loaded = False
        
        if not components_loaded:
            # Use mock components
            self.thermal_manager = MockComponent()
            self.vault_router = MockComponent()
            self.profit_navigator = MockComponent()
            self.entropy_bridge = MockComponent()
            self.dashboard = MockComponent()
            self.logger.info("Mock component suite loaded")
        
        # Initialize Advanced Test Harness
        self.test_harness = AdvancedTestHarness(
            vault=self.vault,
            use_real_components=False  # Use synthetic data for demo
        )
        self.logger.info("Advanced Test Harness initialized")
        
        # Initialize Strategy Execution Mapper
        self.strategy_mapper = StrategyExecutionMapper(
            vault=self.vault,
            vault_router=self.vault_router,
            profit_navigator=self.profit_navigator,
            thermal_manager=self.thermal_manager
        )
        self.logger.info("Strategy Execution Mapper initialized")
        
        # Initialize System Clock Sequencer
        self.clock_sequencer = SystemClockSequencer(
            vault=self.vault,
            strategy_mapper=self.strategy_mapper,
            thermal_manager=self.thermal_manager,
            entropy_bridge=self.entropy_bridge,
            dashboard=self.dashboard
        )
        self.logger.info("System Clock Sequencer initialized")
        
        # Add custom demo tasks to the sequencer
        self._register_demo_tasks()
        
        self.logger.info("Complete system initialization finished")
    
    def _register_demo_tasks(self):
        """Register demo-specific tasks with the clock sequencer"""
        
        # Market simulation task
        self.clock_sequencer.register_task(
            "market_simulation",
            "Simulate realistic market price movements",
            self._simulate_market_tick,
            ScheduleFrequency.EVERY_5_SECONDS,
            critical=True,
            max_runtime=3
        )
        
        # Signal execution task
        self.clock_sequencer.register_task(
            "signal_execution",
            "Execute generated trading signals",
            self._execute_pending_signals,
            ScheduleFrequency.EVERY_15_SECONDS,
            critical=False,
            max_runtime=10
        )
        
        # Demo statistics task
        self.clock_sequencer.register_task(
            "demo_statistics",
            "Update demo statistics and progress",
            self._update_demo_statistics,
            ScheduleFrequency.EVERY_30_SECONDS,
            critical=False,
            max_runtime=5
        )
        
        # Performance showcase task
        self.clock_sequencer.register_task(
            "performance_showcase",
            "Showcase system performance metrics",
            self._showcase_performance,
            ScheduleFrequency.EVERY_MINUTE,
            critical=False,
            max_runtime=10
        )
    
    async def _simulate_market_tick(self):
        """Simulate realistic market price movements"""
        
        if self.live_mode:
            # Simulate more volatile market in live mode
            price_change = np.random.normal(0, self.current_btc_price * 0.002)
            volume_base = 1000000
            volume_multiplier = np.random.uniform(0.5, 2.0)
        else:
            # Calmer simulation for demo
            price_change = np.random.normal(0, self.current_btc_price * 0.001)
            volume_base = 800000
            volume_multiplier = np.random.uniform(0.8, 1.5)
        
        # Update price
        self.current_btc_price += price_change
        self.current_btc_price = max(30000, min(80000, self.current_btc_price))  # Realistic bounds
        
        # Generate volume
        volume = volume_base * volume_multiplier
        
        # Store price history
        self.price_history.append({
            'timestamp': datetime.utcnow(),
            'price': self.current_btc_price,
            'volume': volume
        })
        
        # Keep only last 1000 price points
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
        
        # Generate tick through test harness
        synthetic_tick = self.test_harness.generate_synthetic_tick()
        
        # Override with our simulated price data
        synthetic_tick.btc_price = self.current_btc_price
        synthetic_tick.volume = volume
        
        # Process tick through the system
        tick_result = await self.test_harness.process_synthetic_tick(synthetic_tick)
        
        self.demo_stats['total_vault_ticks'] += 1
        
        # Log significant price movements
        if len(self.price_history) > 1:
            price_change_pct = abs(price_change) / self.current_btc_price
            if price_change_pct > 0.005:  # > 0.5% change
                self.logger.info(f"Significant price movement: ${self.current_btc_price:,.2f} "
                               f"({price_change:+.2f}, {price_change_pct:.2%})")
    
    async def _execute_pending_signals(self):
        """Execute any pending trading signals"""
        
        if len(self.vault.recent_ticks) < 3:
            return  # Need minimum data
        
        # Get latest tick for signal generation
        latest_tick = list(self.vault.recent_ticks)[-1]
        
        # Generate signal
        signal = await self.strategy_mapper.generate_trade_signal(latest_tick)
        
        if signal:
            self.demo_stats['total_signals_generated'] += 1
            
            # Execute the signal
            execution_result = await self.strategy_mapper.execute_signal(signal)
            
            if execution_result.executed:
                self.demo_stats['total_trades_executed'] += 1
                
                # Track strategy performance
                strategy = signal.strategy_type.value
                if strategy not in self.demo_stats['strategy_performance']:
                    self.demo_stats['strategy_performance'][strategy] = {
                        'count': 0, 'total_volume': 0.0, 'avg_confidence': 0.0
                    }
                
                perf = self.demo_stats['strategy_performance'][strategy]
                perf['count'] += 1
                perf['total_volume'] += execution_result.fill_quantity or 0.0
                perf['avg_confidence'] = (
                    (perf['avg_confidence'] * (perf['count'] - 1) + signal.confidence) / perf['count']
                )
                
                self.logger.info(f"Executed {signal.strategy_type.value} {signal.side} signal: "
                               f"${execution_result.fill_price:,.2f} x {execution_result.fill_quantity:.4f}")
            
            else:
                self.logger.warning(f"Failed to execute signal: {execution_result.error_message}")
    
    async def _update_demo_statistics(self):
        """Update comprehensive demo statistics"""
        
        # Get system health from clock sequencer
        system_status = self.clock_sequencer.get_system_status()
        current_health = system_status['system_health_score']
        
        self.demo_stats['system_health_history'].append({
            'timestamp': datetime.utcnow(),
            'health_score': current_health
        })
        
        # Keep only last 100 health readings
        if len(self.demo_stats['system_health_history']) > 100:
            self.demo_stats['system_health_history'] = self.demo_stats['system_health_history'][-100:]
        
        # Check for thermal events
        thermal_state = await self.thermal_manager.get_thermal_state()
        if (thermal_state.get('gpu_temperature', 0) > 80 or 
            thermal_state.get('cpu_temperature', 0) > 75):
            self.demo_stats['thermal_events'] += 1
    
    async def _showcase_performance(self):
        """Showcase comprehensive system performance"""
        
        # Get vault statistics
        vault_stats = self.vault.export_comprehensive_state()
        
        # Get execution statistics
        execution_stats = self.strategy_mapper.get_execution_statistics()
        
        # Get system status
        system_status = self.clock_sequencer.get_system_status()
        
        # Calculate uptime
        if self.demo_stats['start_time']:
            uptime = datetime.utcnow() - self.demo_stats['start_time']
            uptime_str = f"{uptime.total_seconds():.0f}s"
        else:
            uptime_str = "0s"
        
        # Display performance showcase
        self.logger.info("\n" + "="*80)
        self.logger.info("SYSTEM PERFORMANCE SHOWCASE")
        self.logger.info("="*80)
        
        self.logger.info(f"System Uptime: {uptime_str}")
        self.logger.info(f"System Health: {system_status['system_health_score']:.3f}")
        self.logger.info(f"Active Tasks: {system_status['enabled_tasks']}/{system_status['total_tasks']}")
        self.logger.info(f"Avg Cycle Time: {system_status['avg_cycle_time_ms']:.1f}ms")
        
        self.logger.info(f"\nVault Performance:")
        self.logger.info(f"   Total Ticks: {vault_stats['total_ticks']}")
        self.logger.info(f"   Hash Correlations: {vault_stats['hash_correlation_count']}")
        self.logger.info(f"   Recent Anomalies: {len(vault_stats['recent_anomalies'])}")
        self.logger.info(f"   Vault Utilization: {vault_stats['vault_utilization']:.1%}")
        
        if 'error' not in execution_stats:
            self.logger.info(f"\nTrading Performance:")
            self.logger.info(f"   Total Executions: {execution_stats['total_executions']}")
            self.logger.info(f"   Success Rate: {execution_stats['success_rate']:.1%}")
            self.logger.info(f"   Active Signals: {execution_stats['active_signals']}")
        
        self.logger.info(f"\nDemo Statistics:")
        self.logger.info(f"   Signals Generated: {self.demo_stats['total_signals_generated']}")
        self.logger.info(f"   Trades Executed: {self.demo_stats['total_trades_executed']}")
        self.logger.info(f"   Current BTC Price: ${self.current_btc_price:,.2f}")
        self.logger.info(f"   Thermal Events: {self.demo_stats['thermal_events']}")
        
        # Strategy performance breakdown
        if self.demo_stats['strategy_performance']:
            self.logger.info(f"\nStrategy Performance:")
            for strategy, perf in self.demo_stats['strategy_performance'].items():
                self.logger.info(f"   {strategy}: {perf['count']} trades, "
                               f"avg confidence {perf['avg_confidence']:.3f}")
        
        self.logger.info("="*80)
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete integrated system demonstration"""
        
        await self.initialize_system()
        
        self.demo_stats['start_time'] = datetime.utcnow()
        self.logger.info(f"🎬 Starting {self.duration_minutes}-minute integrated system demo")
        
        if self.live_mode:
            self.logger.info("🔴 LIVE MODE: Enhanced market volatility simulation")
        else:
            self.logger.info("🟢 DEMO MODE: Stable market simulation")
        
        # Start the system clock sequencer
        clock_task = asyncio.create_task(self.clock_sequencer.start())
        
        # Run for specified duration
        end_time = datetime.utcnow() + timedelta(minutes=self.duration_minutes)
        
        try:
            while datetime.utcnow() < end_time:
                remaining = (end_time - datetime.utcnow()).total_seconds()
                
                if remaining <= 0:
                    break
                
                # Log progress every 2 minutes
                if int(remaining) % 120 == 0:
                    progress = (self.duration_minutes * 60 - remaining) / (self.duration_minutes * 60) * 100
                    self.logger.info(f"📊 Demo Progress: {progress:.1f}% complete "
                                   f"({remaining/60:.1f} minutes remaining)")
                
                await asyncio.sleep(10)  # Check every 10 seconds
            
            # Stop the clock sequencer
            await self.clock_sequencer.stop()
            
            # Generate final report
            final_report = await self._generate_final_report()
            
            self.logger.info("🎉 Complete integrated system demo finished successfully!")
            
            return final_report
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Demo interrupted by user")
            await self.clock_sequencer.stop()
            return await self._generate_final_report()
        
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            await self.clock_sequencer.stop()
            raise
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        end_time = datetime.utcnow()
        total_runtime = (end_time - self.demo_stats['start_time']).total_seconds()
        
        # Get final system states
        vault_stats = self.vault.export_comprehensive_state()
        execution_stats = self.strategy_mapper.get_execution_statistics()
        system_status = self.clock_sequencer.get_system_status()
        
        # Calculate performance metrics
        signals_per_minute = self.demo_stats['total_signals_generated'] / (total_runtime / 60)
        trades_per_minute = self.demo_stats['total_trades_executed'] / (total_runtime / 60)
        
        # Price movement analysis
        if len(self.price_history) > 1:
            start_price = self.price_history[0]['price']
            end_price = self.price_history[-1]['price']
            price_change_pct = (end_price - start_price) / start_price
        else:
            price_change_pct = 0.0
        
        # System health analysis
        if self.demo_stats['system_health_history']:
            avg_health = sum(h['health_score'] for h in self.demo_stats['system_health_history']) / len(self.demo_stats['system_health_history'])
            min_health = min(h['health_score'] for h in self.demo_stats['system_health_history'])
        else:
            avg_health = 1.0
            min_health = 1.0
        
        final_report = {
            'demo_configuration': {
                'duration_minutes': self.duration_minutes,
                'live_mode': self.live_mode,
                'total_runtime_seconds': total_runtime
            },
            'system_performance': {
                'avg_cycle_time_ms': system_status['avg_cycle_time_ms'],
                'total_cycles': system_status['cycle_count'],
                'final_health_score': system_status['system_health_score'],
                'avg_health_score': avg_health,
                'min_health_score': min_health,
                'active_tasks': system_status['enabled_tasks'],
                'total_tasks': system_status['total_tasks']
            },
            'vault_performance': vault_stats,
            'trading_performance': execution_stats,
            'market_simulation': {
                'start_price': self.price_history[0]['price'] if self.price_history else self.current_btc_price,
                'end_price': self.current_btc_price,
                'price_change_percent': price_change_pct,
                'total_price_updates': len(self.price_history),
                'thermal_events': self.demo_stats['thermal_events']
            },
            'signal_analysis': {
                'total_signals_generated': self.demo_stats['total_signals_generated'],
                'total_trades_executed': self.demo_stats['total_trades_executed'],
                'signals_per_minute': signals_per_minute,
                'trades_per_minute': trades_per_minute,
                'execution_rate': self.demo_stats['total_trades_executed'] / max(self.demo_stats['total_signals_generated'], 1),
                'strategy_performance': self.demo_stats['strategy_performance']
            },
            'recommendations': self._generate_recommendations(vault_stats, execution_stats, system_status)
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"integrated_system_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Final report saved: {report_file}")
        
        return final_report
    
    def _generate_recommendations(self, vault_stats, execution_stats, system_status) -> List[str]:
        """Generate actionable recommendations based on demo results"""
        recommendations = []
        
        # Performance recommendations
        if system_status['avg_cycle_time_ms'] > 100:
            recommendations.append("Consider optimizing task execution - average cycle time is high")
        
        if system_status['system_health_score'] < 0.8:
            recommendations.append("System health score is below optimal - investigate failing tasks")
        
        # Vault recommendations
        if vault_stats['vault_utilization'] > 0.8:
            recommendations.append("Vault utilization is high - consider increasing capacity for production")
        
        if len(vault_stats['recent_anomalies']) > 10:
            recommendations.append("High anomaly count detected - review signal quality")
        
        # Trading recommendations
        if 'error' not in execution_stats:
            if execution_stats['success_rate'] < 0.9:
                recommendations.append("Execution success rate could be improved - check market connectivity")
            
            if execution_stats['total_executions'] == 0:
                recommendations.append("No trades executed - verify signal generation and execution logic")
        
        # Demo-specific recommendations
        if self.demo_stats['total_signals_generated'] < 5:
            recommendations.append("Low signal generation rate - consider adjusting confidence thresholds")
        
        if self.demo_stats['thermal_events'] > 0:
            recommendations.append("Thermal events detected - monitor system cooling")
        
        return recommendations

# CLI interface
async def main():
    """Main entry point for complete integrated system demo"""
    parser = argparse.ArgumentParser(
        description="Complete Integrated Anti-Pole System Demonstration"
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duration in minutes to run the demo (default: 10)'
    )
    parser.add_argument(
        '--live-mode',
        action='store_true',
        help='Enable live mode with enhanced market volatility'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save demo results'
    )
    
    args = parser.parse_args()
    
    # Change to output directory
    import os
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Run the complete demo
    demo = CompleteIntegratedSystemDemo(
        duration_minutes=args.duration,
        live_mode=args.live_mode
    )
    
    final_report = await demo.run_complete_demo()
    
    print("\n" + "🎊 DEMO COMPLETED SUCCESSFULLY! 🎊")
    print(f"📈 Generated {final_report['signal_analysis']['total_signals_generated']} signals")
    print(f"💰 Executed {final_report['signal_analysis']['total_trades_executed']} trades") 
    print(f"🏥 Final system health: {final_report['system_performance']['final_health_score']:.3f}")
    print(f"📊 Vault utilization: {final_report['vault_performance']['vault_utilization']:.1%}")
    
    return final_report

if __name__ == "__main__":
    import numpy as np  # Import here for market simulation
    asyncio.run(main()) 