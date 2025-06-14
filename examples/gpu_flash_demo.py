#!/usr/bin/env python3
"""
GPU Flash Engine v0.5 Demonstration
==================================

This script demonstrates the quantum-coherent GPU flash orchestrator
in action, showing:
- Event-driven architecture integration
- Real-time risk assessment
- Fractal memory and phase coherence
- Configuration management
- State persistence

Usage:
    python examples/gpu_flash_demo.py
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.gpu_flash_engine import GPUFlashEngine, GPUFlashConfig
from core.bus_core import BusCore, BusEvent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_flash_demo')

class MockMarketDataProvider:
    """Mock market data provider for demonstration"""
    
    def __init__(self):
        self.price = 50000.0
        self.volatility = 0.02
        self.trend = 0.001
        self.news_events = []
    
    def generate_price_tick(self):
        """Generate a realistic price tick"""
        # Add trend and random volatility
        change = self.trend + np.random.normal(0, self.volatility)
        self.price *= (1 + change)
        
        # Calculate z-score based on recent volatility
        z_score = abs(change) / self.volatility
        
        # Phase angle based on price momentum
        phase_angle = np.pi + (change * 10)  # Scale change to reasonable phase
        
        return {
            'price': self.price,
            'z_score': z_score,
            'phase_angle': phase_angle,
            'volatility': abs(change)
        }
    
    def inject_news_event(self, severity='medium'):
        """Inject a news event that affects volatility"""
        if severity == 'high':
            self.volatility *= 2.0
            self.trend *= 1.5
        elif severity == 'medium':
            self.volatility *= 1.5
            self.trend *= 1.2
        
        self.news_events.append({
            'timestamp': time.time(),
            'severity': severity
        })
        
        logger.info(f"📰 News event injected: {severity} severity")

class SchwaBot0point5Demo:
    """Demonstration orchestrator for Schwabot v0.5"""
    
    def __init__(self):
        self.bus_core = BusCore()
        self.flash_engine = GPUFlashEngine(bus_core=self.bus_core)
        self.market_data = MockMarketDataProvider()
        
        # Set up event handlers for demonstration
        self.bus_core.register_handler('flash.executed', self._on_flash_executed)
        self.bus_core.register_handler('flash.blocked', self._on_flash_blocked)
        self.bus_core.register_handler('anomaly.detected', self._on_anomaly_detected)
        self.bus_core.register_handler('phase.resonance', self._on_phase_resonance)
        
        self.flash_count = 0
        self.anomaly_count = 0
        
        logger.info("🚀 Schwabot v0.5 Demo initialized")
    
    def _on_flash_executed(self, event: BusEvent):
        """Handle successful flash execution"""
        self.flash_count += 1
        state = event.data['state']
        logger.info(f"⚡ Flash #{self.flash_count} EXECUTED | "
                   f"Risk: {state['risk_entropy']:.3f} | "
                   f"Coherence: {state['coherence_score']:.3f} | "
                   f"Binding: {state['binding_energy']:.1f}")
    
    def _on_flash_blocked(self, event: BusEvent):
        """Handle blocked flash attempt"""
        state = event.data['state']
        logger.warning(f"🚫 Flash BLOCKED | "
                      f"Reason: {state['matrix_state']} | "
                      f"Risk: {state['risk_entropy']:.3f} | "
                      f"Z-score: {state['z_score']:.2f}")
    
    def _on_anomaly_detected(self, event: BusEvent):
        """Handle anomaly detection"""
        self.anomaly_count += 1
        severity = event.data['severity']
        source = event.data['source']
        logger.error(f"🔥 ANOMALY #{self.anomaly_count} | "
                    f"Source: {source} | "
                    f"Severity: {severity}")
    
    def _on_phase_resonance(self, event: BusEvent):
        """Handle phase resonance detection"""
        coherence = event.data['coherence']
        logger.info(f"🌊 Phase RESONANCE detected | "
                   f"Coherence: {coherence:.4f}")
    
    def simulate_market_tick(self):
        """Simulate a single market tick with all the data flows"""
        # Generate market data
        tick_data = self.market_data.generate_price_tick()
        
        # Publish entropy update based on z-score
        self.bus_core.dispatch_event(BusEvent(
            type='entropy.update',
            data={'z_score': tick_data['z_score']},
            timestamp=time.time()
        ))
        
        # Publish phase drift
        self.bus_core.dispatch_event(BusEvent(
            type='phase.drift',
            data={'phase_angle': tick_data['phase_angle']},
            timestamp=time.time()
        ))
        
        # Determine context for flash request
        context = {
            'high_volatility': tick_data['volatility'] > 0.03,
            'news_event': len(self.market_data.news_events) > 0,
            'price': tick_data['price']
        }
        
        # Request flash operation
        self.bus_core.dispatch_event(BusEvent(
            type='flash.request',
            data={
                'z_score': tick_data['z_score'],
                'phase_angle': tick_data['phase_angle'],
                'context': context,
                'request_id': f'tick_{int(time.time() * 1000)}'
            },
            timestamp=time.time()
        ))
        
        return tick_data
    
    def run_demo_scenario(self, duration_seconds=30, tick_interval=0.5):
        """Run a complete demonstration scenario"""
        logger.info(f"🎬 Starting {duration_seconds}s demo scenario...")
        logger.info(f"   Tick interval: {tick_interval}s")
        logger.info(f"   Expected ticks: {int(duration_seconds / tick_interval)}")
        
        start_time = time.time()
        tick_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                tick_count += 1
                
                # Simulate market tick
                tick_data = self.simulate_market_tick()
                
                # Inject news events occasionally
                if tick_count % 10 == 0 and len(self.market_data.news_events) < 2:
                    severity = 'high' if np.random.random() > 0.7 else 'medium'
                    self.market_data.inject_news_event(severity)
                
                # Log current market state every 5 ticks
                if tick_count % 5 == 0:
                    logger.info(f"📊 Tick #{tick_count} | "
                               f"Price: ${tick_data['price']:.0f} | "
                               f"Vol: {tick_data['volatility']:.4f} | "
                               f"Z: {tick_data['z_score']:.2f}")
                
                time.sleep(tick_interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️  Demo interrupted by user")
        
        # Final statistics
        self.print_final_stats(tick_count, duration_seconds)
    
    def print_final_stats(self, tick_count, duration):
        """Print comprehensive final statistics"""
        logger.info("=" * 60)
        logger.info("📈 DEMO COMPLETED - FINAL STATISTICS")
        logger.info("=" * 60)
        
        # Basic counts
        logger.info(f"Total ticks processed: {tick_count}")
        logger.info(f"Flash operations executed: {self.flash_count}")
        logger.info(f"Anomalies detected: {self.anomaly_count}")
        logger.info(f"News events: {len(self.market_data.news_events)}")
        
        # Flash engine statistics
        quantum_stats = self.flash_engine.get_quantum_stats()
        if quantum_stats:
            logger.info(f"Safety rate: {quantum_stats['safety_rate']:.2%}")
            logger.info(f"Mean risk entropy: {quantum_stats['risk']['mean']:.3f}")
            logger.info(f"Current coherence: {quantum_stats['coherence']['current']:.3f}")
            logger.info(f"Max fractal depth: {quantum_stats['fractal']['max_depth']}")
            logger.info(f"Memory usage: {quantum_stats['memory_usage']}")
        
        # Performance metrics
        flash_rate = self.flash_count / duration if duration > 0 else 0
        logger.info(f"Flash rate: {flash_rate:.2f} flashes/second")
        
        logger.info("=" * 60)

def main():
    """Main demonstration function"""
    print("🌌 Schwabot v0.5 GPU Flash Engine Demonstration")
    print("=" * 60)
    print("This demo simulates a quantum-coherent trading environment")
    print("with real-time risk assessment and fractal memory.")
    print()
    
    # Initialize demo
    demo = SchwaBot0point5Demo()
    
    print("Demo scenarios available:")
    print("1. Quick demo (15 seconds)")
    print("2. Standard demo (30 seconds)")
    print("3. Extended demo (60 seconds)")
    print("4. Custom duration")
    print()
    
    try:
        choice = input("Select scenario (1-4): ").strip()
        
        if choice == '1':
            demo.run_demo_scenario(duration_seconds=15, tick_interval=0.3)
        elif choice == '2':
            demo.run_demo_scenario(duration_seconds=30, tick_interval=0.5)
        elif choice == '3':
            demo.run_demo_scenario(duration_seconds=60, tick_interval=0.8)
        elif choice == '4':
            duration = float(input("Enter duration in seconds: "))
            interval = float(input("Enter tick interval in seconds (0.1-2.0): "))
            demo.run_demo_scenario(duration_seconds=duration, tick_interval=interval)
        else:
            print("Invalid choice, running standard demo...")
            demo.run_demo_scenario(duration_seconds=30, tick_interval=0.5)
            
    except (ValueError, KeyboardInterrupt):
        print("\n⏹️  Demo cancelled")
        
    print("\n🎯 Demo complete! Check the logs/ directory for detailed logs.")
    print("📊 State has been persisted to data/gpu_flash_state.json")

if __name__ == '__main__':
    main() 