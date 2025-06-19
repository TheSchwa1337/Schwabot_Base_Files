"""
Quantum BTC Intelligence Core Demonstration
==========================================

This demonstration shows how the enhanced BTC processor integrates with
all mathematical frameworks to create a comprehensive trading intelligence system.

Features Demonstrated:
- Quantum hash correlation with real mining activity
- Altitude-based execution optimization
- Integrated profit vector navigation
- Multivector stability regulation
- Deterministic decision-making framework
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quantum_btc_intelligence_core import (
    QuantumBTCIntelligenceCore,
    create_quantum_btc_intelligence_core,
    ExecutionMode
)

# Setup logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_btc_demo.log')
    ]
)

logger = logging.getLogger(__name__)

class QuantumBTCIntelligenceDemo:
    """
    Comprehensive demonstration of Quantum BTC Intelligence Core capabilities
    """
    
    def __init__(self):
        self.quantum_core: QuantumBTCIntelligenceCore = None
        self.demo_duration = timedelta(minutes=30)  # 30-minute demo
        self.demo_start_time = None
        
    async def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all quantum intelligence features"""
        logger.info("🚀 Starting Quantum BTC Intelligence Core Comprehensive Demo")
        
        try:
            # Initialize the quantum intelligence core
            await self._initialize_quantum_core()
            
            # Run parallel demonstration tasks
            self.demo_start_time = datetime.now()
            
            await asyncio.gather(
                self._demonstrate_hash_correlation(),
                self._demonstrate_altitude_optimization(),
                self._demonstrate_profit_vector_navigation(),
                self._demonstrate_stability_regulation(),
                self._demonstrate_deterministic_decisions(),
                self._monitor_integration_performance(),
                self._demo_time_monitor()
            )
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            await self._cleanup_demo()
    
    async def _initialize_quantum_core(self):
        """Initialize the quantum intelligence core for demonstration"""
        logger.info("🔧 Initializing Quantum BTC Intelligence Core...")
        
        try:
            # Create quantum core with demo configuration
            self.quantum_core = create_quantum_btc_intelligence_core(
                config_path="config/quantum_btc_config.yaml"
            )
            
            logger.info("✅ Quantum BTC Intelligence Core initialized successfully")
            
            # Start the quantum intelligence cycle
            logger.info("🌊 Starting quantum intelligence processing cycle...")
            
            # Note: In a real demo, you would start the full cycle
            # For this demo, we'll simulate the key components
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize quantum core: {e}")
            raise
    
    async def _demonstrate_hash_correlation(self):
        """Demonstrate quantum hash correlation capabilities"""
        logger.info("🔗 Demonstrating Quantum Hash Correlation...")
        
        demo_correlations = []
        
        for i in range(60):  # 1 minute of correlation demos
            try:
                # Simulate hash correlation analysis
                simulated_internal_hash = f"a1b2c3d4e5f6{i:04d}"
                simulated_pool_hash = f"a1b2c3d4e5f6{(i+1):04d}"  # Slightly different
                
                # Calculate correlation using quantum core method
                correlation = self.quantum_core._calculate_hash_correlation(
                    simulated_internal_hash, simulated_pool_hash
                )
                
                # Assess hash quality
                hash_quality = self.quantum_core._assess_internal_hash_quality(
                    simulated_internal_hash
                )
                
                demo_correlations.append({
                    'timestamp': datetime.now().isoformat(),
                    'correlation': correlation,
                    'hash_quality': hash_quality,
                    'internal_hash': simulated_internal_hash[:16] + "...",
                    'pool_hash': simulated_pool_hash[:16] + "..."
                })
                
                # Log significant correlations
                if correlation > 0.7:
                    logger.info(f"🎯 High hash correlation detected: {correlation:.3f}")
                elif correlation > 0.5:
                    logger.info(f"📊 Moderate hash correlation: {correlation:.3f}")
                
                # Update quantum state
                self.quantum_core.quantum_state.pool_hash_correlation = correlation
                self.quantum_core.quantum_state.internal_hash_quality = hash_quality
                
            except Exception as e:
                logger.error(f"Hash correlation demo error: {e}")
            
            await asyncio.sleep(1.0)
        
        # Summary statistics
        avg_correlation = sum(d['correlation'] for d in demo_correlations) / len(demo_correlations)
        max_correlation = max(d['correlation'] for d in demo_correlations)
        
        logger.info(f"📈 Hash Correlation Demo Summary:")
        logger.info(f"   Average Correlation: {avg_correlation:.3f}")
        logger.info(f"   Maximum Correlation: {max_correlation:.3f}")
        logger.info(f"   Total Samples: {len(demo_correlations)}")
    
    async def _demonstrate_altitude_optimization(self):
        """Demonstrate altitude-based execution optimization"""
        logger.info("🛩️ Demonstrating Altitude-Based Execution Optimization...")
        
        altitude_metrics = []
        
        for i in range(40):  # 40 samples over demo period
            try:
                # Simulate varying market conditions
                import numpy as np
                
                # Simulate volume density changes
                base_volume = 1000.0
                volume_variation = np.sin(i * 0.1) * 500 + np.random.normal(0, 100)
                current_volume = max(base_volume + volume_variation, 100.0)
                
                # Simulate price velocity changes
                price_velocity = np.cos(i * 0.15) * 0.02 + np.random.normal(0, 0.005)
                
                # Calculate market altitude using quantum core logic
                volume_density = current_volume / 10000.0
                market_altitude = 1.0 - min(volume_density, 1.0)
                air_density = 1.0 - (market_altitude * 0.33)  # altitude_factor
                
                # Calculate execution pressure
                required_velocity = abs(price_velocity) / (air_density + 0.01)
                execution_pressure = required_velocity * 2.0  # velocity_factor
                pressure_differential = execution_pressure - 1.0  # base_pressure
                
                altitude_metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'volume': current_volume,
                    'price_velocity': price_velocity,
                    'market_altitude': market_altitude,
                    'air_density': air_density,
                    'execution_pressure': execution_pressure,
                    'pressure_differential': pressure_differential
                })
                
                # Update quantum state
                self.quantum_core.quantum_state.execution_pressure = execution_pressure
                self.quantum_core.quantum_state.optimal_altitude = market_altitude
                self.quantum_core.quantum_state.pressure_differential = pressure_differential
                
                # Log significant altitude changes
                if market_altitude > 0.8:
                    logger.info(f"🛩️ High altitude detected: {market_altitude:.3f} (thin market)")
                elif abs(pressure_differential) > 0.5:
                    logger.info(f"⚡ Significant pressure differential: {pressure_differential:.3f}")
                
            except Exception as e:
                logger.error(f"Altitude optimization demo error: {e}")
            
            await asyncio.sleep(1.5)
        
        # Analysis summary
        avg_altitude = sum(m['market_altitude'] for m in altitude_metrics) / len(altitude_metrics)
        max_pressure = max(abs(m['pressure_differential']) for m in altitude_metrics)
        
        logger.info(f"📊 Altitude Optimization Demo Summary:")
        logger.info(f"   Average Market Altitude: {avg_altitude:.3f}")
        logger.info(f"   Maximum Pressure Differential: {max_pressure:.3f}")
        logger.info(f"   Pressure Events: {sum(1 for m in altitude_metrics if abs(m['pressure_differential']) > 0.3)}")
    
    async def _demonstrate_profit_vector_navigation(self):
        """Demonstrate integrated profit vector navigation"""
        logger.info("🧭 Demonstrating Profit Vector Navigation...")
        
        profit_vectors = []
        
        for i in range(30):  # 30 samples over demo period
            try:
                # Simulate market state for profit calculation
                import numpy as np
                
                base_price = 50000.0
                price_change = np.sin(i * 0.2) * 2000 + np.random.normal(0, 500)
                current_price = base_price + price_change
                
                base_volume = 1500.0
                volume_change = np.cos(i * 0.15) * 300 + np.random.normal(0, 100)
                current_volume = max(base_volume + volume_change, 100.0)
                
                # Update profit navigator
                profit_vector = self.quantum_core.profit_navigator.update_market_state(
                    current_price=current_price,
                    current_volume=current_volume,
                    timestamp=datetime.now()
                )
                
                # Calculate trajectory stability
                trajectory_stability = self.quantum_core._calculate_profit_trajectory_stability()
                
                vector_data = {
                    'timestamp': datetime.now().isoformat(),
                    'price': current_price,
                    'volume': current_volume,
                    'vector_magnitude': profit_vector.magnitude if profit_vector else 0.0,
                    'vector_confidence': profit_vector.confidence if profit_vector else 0.0,
                    'vector_direction': profit_vector.direction if profit_vector else 0,
                    'trajectory_stability': trajectory_stability
                }
                
                profit_vectors.append(vector_data)
                
                # Update quantum state
                self.quantum_core.quantum_state.primary_vector_magnitude = vector_data['vector_magnitude']
                self.quantum_core.quantum_state.vector_confidence = vector_data['vector_confidence']
                self.quantum_core.quantum_state.profit_trajectory_stability = trajectory_stability
                
                # Log significant vectors
                if vector_data['vector_magnitude'] > 0.3:
                    direction_str = "LONG" if vector_data['vector_direction'] > 0 else "SHORT" if vector_data['vector_direction'] < 0 else "HOLD"
                    logger.info(f"🎯 Strong profit vector: {direction_str} magnitude={vector_data['vector_magnitude']:.3f}")
                
            except Exception as e:
                logger.error(f"Profit vector demo error: {e}")
            
            await asyncio.sleep(2.0)
        
        # Vector analysis summary
        strong_vectors = [v for v in profit_vectors if v['vector_magnitude'] > 0.2]
        avg_confidence = sum(v['vector_confidence'] for v in profit_vectors) / len(profit_vectors)
        
        logger.info(f"🧭 Profit Vector Demo Summary:")
        logger.info(f"   Strong Vectors Detected: {len(strong_vectors)}")
        logger.info(f"   Average Confidence: {avg_confidence:.3f}")
        logger.info(f"   Vector Success Rate: {len(strong_vectors)/len(profit_vectors)*100:.1f}%")
    
    async def _demonstrate_stability_regulation(self):
        """Demonstrate multivector stability regulation"""
        logger.info("⚖️ Demonstrating Multivector Stability Regulation...")
        
        stability_metrics = []
        
        for i in range(20):  # 20 samples over demo period
            try:
                # Simulate system metrics
                import numpy as np
                
                # Simulate varying system loads
                cpu_usage = 60.0 + np.sin(i * 0.3) * 20 + np.random.normal(0, 5)
                memory_usage = 6.0 + np.cos(i * 0.2) * 2 + np.random.normal(0, 0.5)
                active_processes = int(30 + np.random.normal(0, 10))
                
                # Calculate stability metrics using quantum core logic
                hash_coherence = self.quantum_core.quantum_state.pool_hash_correlation
                pressure_coherence = 1.0 - abs(self.quantum_core.quantum_state.pressure_differential)
                profit_coherence = self.quantum_core.quantum_state.vector_confidence
                
                multivector_coherence = np.mean([hash_coherence, pressure_coherence, profit_coherence])
                
                # System stability calculation
                cpu_stability = 1.0 - (cpu_usage / 100.0)
                memory_stability = 1.0 - (memory_usage / 16.0)
                process_stability = min(1.0, 10.0 / max(active_processes, 1))
                
                system_stability = np.mean([cpu_stability, memory_stability, process_stability])
                
                # Resource optimization
                cpu_efficiency = max(0.0, 1.0 - abs(cpu_usage - 70.0) / 70.0)
                memory_efficiency = max(0.0, 1.0 - memory_usage / 12.8)
                process_efficiency = max(0.0, 1.0 - active_processes / 100.0)
                
                resource_optimization = np.mean([cpu_efficiency, memory_efficiency, process_efficiency])
                
                stability_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'active_processes': active_processes,
                    'multivector_coherence': multivector_coherence,
                    'system_stability': system_stability,
                    'resource_optimization': resource_optimization
                }
                
                stability_metrics.append(stability_data)
                
                # Update quantum state
                self.quantum_core.quantum_state.multivector_coherence = multivector_coherence
                self.quantum_core.quantum_state.system_stability_index = system_stability
                self.quantum_core.quantum_state.resource_optimization_level = resource_optimization
                
                # Log stability events
                if system_stability < 0.7:
                    logger.warning(f"⚠️ Low system stability: {system_stability:.3f}")
                elif multivector_coherence > 0.8:
                    logger.info(f"✅ High multivector coherence: {multivector_coherence:.3f}")
                
            except Exception as e:
                logger.error(f"Stability regulation demo error: {e}")
            
            await asyncio.sleep(3.0)
        
        # Stability analysis
        avg_stability = sum(s['system_stability'] for s in stability_metrics) / len(stability_metrics)
        low_stability_events = sum(1 for s in stability_metrics if s['system_stability'] < 0.7)
        
        logger.info(f"⚖️ Stability Regulation Demo Summary:")
        logger.info(f"   Average System Stability: {avg_stability:.3f}")
        logger.info(f"   Low Stability Events: {low_stability_events}")
        logger.info(f"   Stability Success Rate: {(len(stability_metrics)-low_stability_events)/len(stability_metrics)*100:.1f}%")
    
    async def _demonstrate_deterministic_decisions(self):
        """Demonstrate deterministic decision-making framework"""
        logger.info("🎯 Demonstrating Deterministic Decision Framework...")
        
        decisions = []
        
        for i in range(15):  # 15 decision cycles over demo period
            try:
                # Calculate deterministic confidence using quantum core logic
                hash_factor = min(self.quantum_core.quantum_state.pool_hash_correlation, 1.0)
                pressure_factor = min(abs(self.quantum_core.quantum_state.pressure_differential), 1.0)
                vector_factor = min(self.quantum_core.quantum_state.primary_vector_magnitude, 1.0)
                stability_factor = self.quantum_core.quantum_state.system_stability_index
                
                deterministic_confidence = np.sqrt(
                    hash_factor * pressure_factor * vector_factor * stability_factor
                )
                
                mathematical_certainty = (
                    deterministic_confidence * 
                    self.quantum_core.quantum_state.multivector_coherence * 
                    self.quantum_core.quantum_state.resource_optimization_level
                )
                
                execution_readiness = min(deterministic_confidence + mathematical_certainty, 1.0)
                
                # Create decision using quantum core method
                decision = await self.quantum_core._create_quantum_execution_decision()
                
                decision_data = {
                    'timestamp': datetime.now().isoformat(),
                    'deterministic_confidence': deterministic_confidence,
                    'mathematical_certainty': mathematical_certainty,
                    'execution_readiness': execution_readiness,
                    'should_execute': decision.should_execute,
                    'position_size': decision.position_size,
                    'confidence_level': decision.confidence_level,
                    'execution_mode': decision.execution_mode.value
                }
                
                decisions.append(decision_data)
                
                # Update quantum state
                self.quantum_core.quantum_state.deterministic_confidence = deterministic_confidence
                self.quantum_core.quantum_state.mathematical_certainty = mathematical_certainty
                self.quantum_core.quantum_state.execution_readiness = execution_readiness
                
                # Log significant decisions
                if decision.should_execute:
                    logger.info(f"🚀 EXECUTE decision: confidence={decision.confidence_level:.3f}, size={decision.position_size:.3f}")
                elif execution_readiness > 0.7:
                    logger.info(f"⏳ High readiness (HOLD): {execution_readiness:.3f}")
                
            except Exception as e:
                logger.error(f"Decision framework demo error: {e}")
            
            await asyncio.sleep(4.0)
        
        # Decision analysis
        execute_decisions = [d for d in decisions if d['should_execute']]
        avg_readiness = sum(d['execution_readiness'] for d in decisions) / len(decisions)
        
        logger.info(f"🎯 Decision Framework Demo Summary:")
        logger.info(f"   Execute Decisions: {len(execute_decisions)}/{len(decisions)}")
        logger.info(f"   Average Execution Readiness: {avg_readiness:.3f}")
        logger.info(f"   Decision Success Rate: {len(execute_decisions)/len(decisions)*100:.1f}%")
    
    async def _monitor_integration_performance(self):
        """Monitor overall integration performance"""
        logger.info("📊 Monitoring Integration Performance...")
        
        while datetime.now() - self.demo_start_time < self.demo_duration:
            try:
                # Get quantum state summary
                state_summary = self.quantum_core.get_quantum_state_summary()
                
                # Log performance metrics every 5 minutes
                if (datetime.now() - self.demo_start_time).seconds % 300 == 0:
                    logger.info("📈 Integration Performance Report:")
                    logger.info(f"   System Health: {state_summary['system_health']}")
                    logger.info(f"   Hash Correlation Trend: {state_summary['hash_correlation_trend']:.3f}")
                    logger.info(f"   Recent Decisions: {state_summary['recent_decisions']}")
                
                # Determine optimal execution mode
                optimal_mode = self.quantum_core._determine_optimal_execution_mode()
                self.quantum_core.quantum_state.execution_mode = optimal_mode
                
            except Exception as e:
                logger.error(f"Integration monitoring error: {e}")
            
            await asyncio.sleep(30.0)  # Check every 30 seconds
    
    async def _demo_time_monitor(self):
        """Monitor demo time and provide updates"""
        while datetime.now() - self.demo_start_time < self.demo_duration:
            elapsed = datetime.now() - self.demo_start_time
            remaining = self.demo_duration - elapsed
            
            if remaining.seconds % 300 == 0:  # Every 5 minutes
                logger.info(f"⏰ Demo Progress: {elapsed.seconds//60} minutes elapsed, {remaining.seconds//60} minutes remaining")
            
            await asyncio.sleep(60.0)  # Check every minute
        
        logger.info("⏰ Demo time completed!")
    
    async def _cleanup_demo(self):
        """Clean up demo resources"""
        logger.info("🧹 Cleaning up demo resources...")
        
        try:
            if self.quantum_core:
                await self.quantum_core.shutdown()
            
            # Generate demo summary report
            await self._generate_demo_report()
            
            logger.info("✅ Demo cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Demo cleanup error: {e}")
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        logger.info("📋 Generating Demo Report...")
        
        try:
            # Get final quantum state
            final_state = self.quantum_core.get_quantum_state_summary()
            
            # Get execution history
            execution_history = self.quantum_core.get_execution_decision_history(20)
            
            demo_report = {
                'demo_info': {
                    'start_time': self.demo_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.demo_start_time).seconds // 60
                },
                'final_quantum_state': final_state,
                'execution_decisions': execution_history,
                'performance_summary': {
                    'total_decisions': len(execution_history),
                    'execute_decisions': sum(1 for d in execution_history if d['should_execute']),
                    'average_confidence': sum(d['confidence_level'] for d in execution_history) / max(len(execution_history), 1),
                    'execution_modes_used': list(set(d['execution_mode'] for d in execution_history))
                }
            }
            
            # Save report to file
            report_file = f"quantum_btc_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(demo_report, f, indent=2)
            
            logger.info(f"📋 Demo report saved to: {report_file}")
            logger.info("🎉 Quantum BTC Intelligence Core Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo report generation error: {e}")


# Demo execution function
async def run_quantum_btc_demo():
    """Run the quantum BTC intelligence demo"""
    demo = QuantumBTCIntelligenceDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Run the comprehensive demo
    try:
        asyncio.run(run_quantum_btc_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise 