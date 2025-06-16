#!/usr/bin/env python3
"""
Complete Anti-Pole System Demo v1.0
===================================

Comprehensive demonstration of the Anti-Pole Tesseract Visualizer Bridge
with full Schwabot integration, real-time visualization, and live trading signals.

Features:
- Real-time market data simulation
- Quantum Anti-Pole Engine processing
- Entropy Bridge integration
- Dashboard visualization server
- Performance monitoring
- Trading signal generation
- Complete system health monitoring

Usage:
    python demo_complete_antipole_system.py
"""

import asyncio
import logging
import time
import json
import math
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from dataclasses import asdict

# Core imports
try:
    from core.quantum_antipole_engine import QuantumAntiPoleEngine, QAConfig
    from core.entropy_bridge import EntropyBridge, EntropyBridgeConfig
    from core.dashboard_integration import DashboardIntegration, DashboardConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class BTCMarketSimulator:
    """
    Advanced BTC market data simulator with realistic patterns
    """
    
    def __init__(self, base_price: float = 45000.0, base_volume: float = 1000000.0):
        self.base_price = base_price
        self.base_volume = base_volume
        self.current_price = base_price
        self.current_volume = base_volume
        
        # Market regime state
        self.regime = "NORMAL"  # BULL, BEAR, VOLATILE, NORMAL
        self.regime_duration = 0
        self.tick_count = 0
        
        # Trend parameters
        self.trend_direction = 1  # 1 for up, -1 for down
        self.trend_strength = 0.5
        self.volatility_multiplier = 1.0
        
        # Market events
        self.last_major_event = 0
        self.event_cooldown = 100  # Ticks between major events
        
    def next_tick(self) -> Dict:
        """Generate next realistic market tick"""
        self.tick_count += 1
        
        # Update market regime periodically
        if self.tick_count % 50 == 0:
            self._update_market_regime()
        
        # Generate price movement
        price_change = self._generate_price_movement()
        self.current_price = max(self.current_price + price_change, 1000)  # Minimum price
        
        # Generate volume
        volume_change = self._generate_volume_movement()
        self.current_volume = max(self.current_volume + volume_change, 10000)  # Minimum volume
        
        # Occasional market events
        if self.tick_count - self.last_major_event > self.event_cooldown:
            if np.random.random() < 0.05:  # 5% chance of major event
                self._generate_market_event()
        
        return {
            'price': self.current_price,
            'volume': self.current_volume,
            'timestamp': datetime.utcnow(),
            'regime': self.regime,
            'trend_direction': self.trend_direction,
            'volatility': self.volatility_multiplier
        }
    
    def _generate_price_movement(self) -> float:
        """Generate realistic price movement based on current regime"""
        base_movement = 0
        
        if self.regime == "BULL":
            base_movement = np.random.normal(2.0, 8.0)  # Upward bias
        elif self.regime == "BEAR":
            base_movement = np.random.normal(-2.0, 8.0)  # Downward bias
        elif self.regime == "VOLATILE":
            base_movement = np.random.normal(0, 15.0)  # High volatility, no bias
        else:  # NORMAL
            base_movement = np.random.normal(0, 5.0)  # Moderate volatility
        
        # Add trend component
        trend_component = self.trend_direction * self.trend_strength * np.random.normal(0.5, 0.2)
        
        # Add mean reversion component
        distance_from_base = (self.current_price - self.base_price) / self.base_price
        mean_reversion = -distance_from_base * 10 * np.random.random()
        
        # Combine components
        total_movement = base_movement + trend_component + mean_reversion
        
        return total_movement * self.volatility_multiplier
    
    def _generate_volume_movement(self) -> float:
        """Generate volume movement correlated with price volatility"""
        # Base volume change
        base_volume_change = np.random.normal(0, self.base_volume * 0.1)
        
        # Volume spike during high volatility
        if self.regime == "VOLATILE":
            if np.random.random() < 0.3:  # 30% chance of volume spike
                base_volume_change += np.random.normal(self.base_volume * 0.2, self.base_volume * 0.1)
        
        # Mean reversion for volume
        distance_from_base = (self.current_volume - self.base_volume) / self.base_volume
        volume_reversion = -distance_from_base * self.base_volume * 0.1
        
        return base_volume_change + volume_reversion
    
    def _update_market_regime(self):
        """Update market regime periodically"""
        self.regime_duration += 1
        
        # Regime transition probabilities
        if self.regime_duration > 20:  # Minimum regime duration
            transition_prob = 0.1  # 10% chance to change regime
            
            if np.random.random() < transition_prob:
                regimes = ["BULL", "BEAR", "VOLATILE", "NORMAL"]
                new_regime = np.random.choice([r for r in regimes if r != self.regime])
                
                print(f"📊 Market regime change: {self.regime} → {new_regime}")
                self.regime = new_regime
                self.regime_duration = 0
                
                # Update parameters based on new regime
                if new_regime == "BULL":
                    self.trend_direction = 1
                    self.trend_strength = 0.8
                    self.volatility_multiplier = 0.7
                elif new_regime == "BEAR":
                    self.trend_direction = -1
                    self.trend_strength = 0.8
                    self.volatility_multiplier = 0.8
                elif new_regime == "VOLATILE":
                    self.trend_direction = np.random.choice([-1, 1])
                    self.trend_strength = 0.3
                    self.volatility_multiplier = 1.5
                else:  # NORMAL
                    self.trend_direction = np.random.choice([-1, 1])
                    self.trend_strength = 0.4
                    self.volatility_multiplier = 1.0
    
    def _generate_market_event(self):
        """Generate major market event"""
        events = [
            {"name": "Institutional Buy", "price_impact": 500, "volume_impact": 0.5},
            {"name": "Whale Dump", "price_impact": -800, "volume_impact": 0.8},
            {"name": "News Spike", "price_impact": 300, "volume_impact": 0.6},
            {"name": "Regulatory FUD", "price_impact": -400, "volume_impact": 0.4},
            {"name": "Technical Breakout", "price_impact": 600, "volume_impact": 0.3}
        ]
        
        event = np.random.choice(events)
        print(f"🚨 Market Event: {event['name']}")
        
        # Apply event impact
        self.current_price += event['price_impact'] * np.random.uniform(0.5, 1.5)
        self.current_volume *= (1 + event['volume_impact'])
        
        # Increase volatility temporarily
        self.volatility_multiplier *= 1.5
        
        self.last_major_event = self.tick_count

class SystemHealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.tick_count = 0
        self.total_processing_time = 0.0
        self.max_processing_time = 0.0
        self.min_processing_time = float('inf')
        
        self.error_count = 0
        self.last_errors = []
        
        self.performance_history = []
        
    def record_tick(self, processing_time_ms: float, errors: List[str] = None):
        """Record tick performance"""
        self.tick_count += 1
        self.total_processing_time += processing_time_ms
        self.max_processing_time = max(self.max_processing_time, processing_time_ms)
        self.min_processing_time = min(self.min_processing_time, processing_time_ms)
        
        if errors:
            self.error_count += len(errors)
            self.last_errors.extend(errors[-5:])  # Keep last 5 errors
            self.last_errors = self.last_errors[-10:]  # Limit to 10 total
        
        # Store performance history (last 100 ticks)
        self.performance_history.append(processing_time_ms)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        uptime = time.time() - self.start_time
        avg_processing_time = (
            self.total_processing_time / max(self.tick_count, 1)
        )
        
        recent_performance = (
            np.mean(self.performance_history[-20:]) 
            if len(self.performance_history) >= 20 
            else avg_processing_time
        )
        
        # Health scoring
        health_score = 100.0
        
        # Deduct for slow performance
        if avg_processing_time > 50:
            health_score -= min(30, (avg_processing_time - 50) / 2)
        
        # Deduct for errors
        error_rate = self.error_count / max(self.tick_count, 1)
        health_score -= min(40, error_rate * 1000)
        
        # Deduct for inconsistent performance
        if len(self.performance_history) > 10:
            performance_std = np.std(self.performance_history)
            if performance_std > 20:
                health_score -= min(20, (performance_std - 20) / 5)
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'status': (
                'EXCELLENT' if health_score > 90 else
                'GOOD' if health_score > 75 else
                'FAIR' if health_score > 50 else
                'POOR'
            ),
            'uptime_seconds': uptime,
            'total_ticks': self.tick_count,
            'avg_processing_time_ms': avg_processing_time,
            'recent_processing_time_ms': recent_performance,
            'max_processing_time_ms': self.max_processing_time,
            'min_processing_time_ms': self.min_processing_time if self.min_processing_time != float('inf') else 0,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'last_errors': self.last_errors[-5:],
            'throughput_per_second': self.tick_count / max(uptime, 1)
        }

class CompleteAntiPoleDemo:
    """Main demo orchestrator"""
    
    def __init__(self):
        self.market_simulator = BTCMarketSimulator()
        self.health_monitor = SystemHealthMonitor()
        
        # System components
        self.quantum_engine: Optional[QuantumAntiPoleEngine] = None
        self.entropy_bridge: Optional[EntropyBridge] = None
        self.dashboard: Optional[DashboardIntegration] = None
        
        # Control flags
        self.running = False
        self.shutdown_requested = False
        
        # Statistics
        self.total_opportunities = 0
        self.profitable_signals = 0
        self.portfolio_value = 100000.0  # Starting portfolio
        
        # Setup logging
        self.setup_logging()
        self.log = logging.getLogger("antipole.demo")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"antipole_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        
        # Reduce noise from some loggers
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
    
    async def initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing Complete Anti-Pole System...")
        
        try:
            # Initialize Quantum Anti-Pole Engine
            qa_config = QAConfig(
                use_gpu=True,  # Try GPU first, fallback to CPU
                field_size=64,
                tick_window=256,
                pole_order=12,
                debug_mode=False,
                use_entropy_tracker=True,
                use_thermal_manager=True,
                use_ferris_wheel=True
            )
            
            self.quantum_engine = QuantumAntiPoleEngine(qa_config)
            print("✅ Quantum Anti-Pole Engine initialized")
            
            # Initialize Entropy Bridge
            entropy_config = EntropyBridgeConfig(
                websocket_port=8767,
                json_export_path="data/live_entropy.json",
                use_existing_entropy=True,
                use_quantum_engine=False  # We'll manually connect
            )
            
            self.entropy_bridge = EntropyBridge(entropy_config)
            self.entropy_bridge.quantum_engine = self.quantum_engine  # Manual connection
            print("✅ Entropy Bridge initialized")
            
            # Initialize Dashboard Integration
            dashboard_config = DashboardConfig(
                host="0.0.0.0",  # Allow external connections
                port=8768,
                update_frequency=15.0,  # 15 FPS
                enable_entropy_bridge=False,  # We'll manually connect
                enable_profit_navigator=False,
                enable_thermal_monitoring=True
            )
            
            self.dashboard = DashboardIntegration(dashboard_config)
            self.dashboard.entropy_bridge = self.entropy_bridge  # Manual connection
            print("✅ Dashboard Integration initialized")
            
            # Start dashboard server
            await self.dashboard.start_server()
            
            # Start entropy bridge WebSocket server
            await self.entropy_bridge.start_websocket_server()
            
            print("🌐 All servers started successfully!")
            print(f"📊 Dashboard: http://localhost:8768/")
            print(f"🔌 WebSocket: ws://localhost:8768/ws/live-data")
            print(f"📈 Entropy API: http://localhost:8768/api/current-data")
            
            return True
            
        except Exception as e:
            self.log.error(f"System initialization failed: {e}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def main_processing_loop(self):
        """Main processing loop"""
        print("\n🔄 Starting main processing loop...")
        print("📈 Real-time Anti-Pole analysis beginning...")
        print("🎯 Press Ctrl+C to stop\n")
        
        tick_interval = 0.1  # 10 FPS
        last_status_update = 0
        status_interval = 5.0  # Status update every 5 seconds
        
        while self.running and not self.shutdown_requested:
            loop_start = time.perf_counter()
            errors = []
            
            try:
                # Generate market tick
                market_data = self.market_simulator.next_tick()
                
                # Process through entropy bridge (which uses quantum engine)
                start_time = time.perf_counter()
                flow_data = await self.entropy_bridge.process_market_tick(
                    market_data['price'],
                    market_data['volume'],
                    market_data['timestamp']
                )
                processing_time = (time.perf_counter() - start_time) * 1000
                
                # Process through dashboard integration
                dashboard_data = await self.dashboard.process_market_tick(
                    market_data['price'],
                    market_data['volume'],
                    market_data['timestamp']
                )
                
                # Update portfolio simulation
                self._update_portfolio_simulation(flow_data, market_data)
                
                # Record performance
                self.health_monitor.record_tick(processing_time, errors)
                
                # Periodic status updates
                current_time = time.time()
                if current_time - last_status_update >= status_interval:
                    self._print_status_update(market_data, flow_data, dashboard_data)
                    last_status_update = current_time
                
            except Exception as e:
                error_msg = f"Processing error: {e}"
                errors.append(error_msg)
                self.log.error(error_msg)
            
            # Maintain tick rate
            loop_time = time.perf_counter() - loop_start
            sleep_time = max(0, tick_interval - loop_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    def _update_portfolio_simulation(self, flow_data, market_data):
        """Update simulated portfolio based on signals"""
        # Simple portfolio simulation
        if flow_data.recommendation == "STRONG_BUY" and flow_data.confidence > 0.7:
            # Simulate buy order
            buy_amount = self.portfolio_value * 0.1  # 10% of portfolio
            self.portfolio_value -= buy_amount  # Spend cash
            self.total_opportunities += 1
            
            # Simplified profit calculation (assuming good timing)
            if np.random.random() < 0.6:  # 60% win rate for strong signals
                profit = buy_amount * 0.02  # 2% profit
                self.portfolio_value += buy_amount + profit
                self.profitable_signals += 1
            else:
                loss = buy_amount * 0.01  # 1% loss
                self.portfolio_value += buy_amount - loss
        
        elif flow_data.recommendation == "STRONG_SELL" and flow_data.confidence > 0.7:
            # Simulate sell order (simplified - assume we have position)
            self.total_opportunities += 1
            
            if np.random.random() < 0.55:  # 55% win rate for sell signals
                profit = self.portfolio_value * 0.015  # 1.5% profit
                self.portfolio_value += profit
                self.profitable_signals += 1
    
    def _print_status_update(self, market_data, flow_data, dashboard_data):
        """Print comprehensive status update"""
        health = self.health_monitor.get_health_report()
        
        print(f"\n{'='*80}")
        print(f"📊 ANTI-POLE TESSERACT VISUALIZER - STATUS UPDATE")
        print(f"{'='*80}")
        
        # Market data
        print(f"💰 Market Data:")
        print(f"   Price: ${market_data['price']:,.2f}")
        print(f"   Volume: {market_data['volume']:,.0f}")
        print(f"   Regime: {market_data['regime']}")
        print(f"   Trend: {'📈' if market_data['trend_direction'] > 0 else '📉'}")
        
        # Entropy metrics
        print(f"\n🌀 Entropy Analysis:")
        print(f"   Hash Entropy: {flow_data.hash_entropy:.3f}")
        print(f"   Quantum Entropy: {flow_data.quantum_entropy:.3f}")
        print(f"   Combined: {flow_data.combined_entropy:.3f}")
        print(f"   Tier: {flow_data.entropy_tier}")
        
        # Anti-pole metrics
        print(f"\n⚛️  Anti-Pole Metrics:")
        print(f"   AP-RSI: {flow_data.ap_rsi:.1f}")
        print(f"   Coherence: {flow_data.coherence:.3f}")
        print(f"   Poles: {flow_data.pole_count}")
        print(f"   Vector Strength: {flow_data.vector_strength:.3f}")
        
        # Trading signals
        signal_emoji = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG_SELL': '💥'
        }
        
        print(f"\n🎯 Trading Signals:")
        print(f"   Recommendation: {signal_emoji.get(flow_data.recommendation, '❓')} {flow_data.recommendation}")
        print(f"   Signal Strength: {flow_data.signal_strength:+.3f}")
        print(f"   Confidence: {flow_data.confidence:.1%}")
        
        # Portfolio simulation
        win_rate = (self.profitable_signals / max(self.total_opportunities, 1)) * 100
        portfolio_change = ((self.portfolio_value - 100000) / 100000) * 100
        
        print(f"\n💼 Portfolio Simulation:")
        print(f"   Value: ${self.portfolio_value:,.2f}")
        print(f"   P&L: {portfolio_change:+.2f}%")
        print(f"   Opportunities: {self.total_opportunities}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        # System performance
        print(f"\n⚡ System Performance:")
        print(f"   Health: {health['status']} ({health['health_score']:.1f}/100)")
        print(f"   Processing: {health['avg_processing_time_ms']:.1f}ms avg")
        print(f"   Throughput: {health['throughput_per_second']:.1f} ticks/sec")
        print(f"   Uptime: {health['uptime_seconds']:.0f}s")
        print(f"   Errors: {health['error_count']}")
        
        # Dashboard status
        dashboard_status = dashboard_data.system_health if dashboard_data else {}
        print(f"\n🌐 Dashboard Status:")
        print(f"   Entropy Bridge: {'✅' if dashboard_status.get('entropy_bridge_active') else '❌'}")
        print(f"   WebSocket Clients: {len(self.dashboard.websocket_connections) if self.dashboard else 0}")
        print(f"   API Server: {'✅' if dashboard_status.get('api_server_running') else '❌'}")
        
        print(f"{'='*80}\n")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(sig, frame):
            print(f"\n🛑 Received signal {sig}. Initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown_system(self):
        """Graceful system shutdown"""
        print("\n🔄 Shutting down Anti-Pole system...")
        
        # Stop main loop
        self.running = False
        
        # Shutdown components
        if self.dashboard:
            await self.dashboard.stop_server()
            print("✅ Dashboard server stopped")
        
        if self.entropy_bridge:
            await self.entropy_bridge.stop_websocket_server()
            print("✅ Entropy bridge stopped")
        
        if self.quantum_engine:
            self.quantum_engine.shutdown()
            print("✅ Quantum engine stopped")
        
        # Final health report
        final_health = self.health_monitor.get_health_report()
        print(f"\n📊 Final Performance Report:")
        print(f"   Total ticks processed: {final_health['total_ticks']}")
        print(f"   Average processing time: {final_health['avg_processing_time_ms']:.2f}ms")
        print(f"   System uptime: {final_health['uptime_seconds']:.1f}s")
        print(f"   Final health score: {final_health['health_score']:.1f}/100")
        print(f"   Portfolio final value: ${self.portfolio_value:,.2f}")
        
        portfolio_return = ((self.portfolio_value - 100000) / 100000) * 100
        print(f"   Portfolio return: {portfolio_return:+.2f}%")
        
        print("\n✅ Anti-Pole system shutdown complete!")
    
    async def run(self):
        """Main entry point"""
        print("🌟 Complete Anti-Pole Tesseract Visualizer Bridge Demo")
        print("=" * 60)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Initialize system
        if not await self.initialize_system():
            return 1
        
        try:
            # Start main processing
            self.running = True
            await self.main_processing_loop()
            
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt received")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            self.log.error(f"Unexpected error: {e}")
        finally:
            await self.shutdown_system()
        
        return 0

async def main():
    """Main entry point"""
    demo = CompleteAntiPoleDemo()
    return await demo.run()

if __name__ == "__main__":
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║    🌟 ANTI-POLE TESSERACT VISUALIZER BRIDGE v1.0 🌟                        ║
    ║                                                                               ║
    ║    Advanced Quantum Financial Navigation System                               ║
    ║    Real-time entropy analysis • Anti-pole theory • 4D visualization         ║
    ║                                                                               ║
    ║    Integration Components:                                                    ║
    ║    ⚛️  Quantum Anti-Pole Engine                                              ║
    ║    🌉 Entropy Bridge                                                         ║
    ║    📊 Dashboard Integration                                                  ║
    ║    🔥 Thermal Protection                                                     ║
    ║    🎡 Ferris Wheel Strategy                                                  ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    try:
        # Run the demo
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 