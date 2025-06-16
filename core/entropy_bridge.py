"""
Entropy Bridge v1.0
===================

Bridge component connecting the Quantum Anti-Pole Engine with existing
Schwabot entropy systems and dashboard visualization.

Provides:
- Real-time entropy data flow
- Integration with existing entropy_tracker.py
- JSON export for dashboard consumption
- WebSocket streaming for live updates
"""

import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import websockets

# Core Schwabot imports
try:
    from .entropy_tracker import EntropyTracker
    from .quantum_antipole_engine import QuantumAntiPoleEngine, QAConfig
    from .hash_profit_matrix import HashProfitMatrix
    from .ferris_wheel_scheduler import FerrisWheelScheduler
except ImportError:
    # Fallback for standalone usage
    EntropyTracker = None
    QuantumAntiPoleEngine = None
    QAConfig = None
    HashProfitMatrix = None
    FerrisWheelScheduler = None

@dataclass
class EntropyFlowData:
    """Complete entropy flow data package"""
    timestamp: datetime
    price: float
    volume: float
    
    # Core entropy metrics
    hash_entropy: float
    quantum_entropy: float
    combined_entropy: float
    entropy_tier: str
    
    # Anti-pole metrics  
    ap_rsi: float
    coherence: float
    pole_count: int
    vector_strength: float
    
    # Trading signals
    signal_strength: float
    recommendation: str
    confidence: float
    
    # Performance
    computation_time_ms: float

@dataclass
class EntropyBridgeConfig:
    """Configuration for entropy bridge"""
    # Data retention
    history_size: int = 1000
    websocket_port: int = 8767
    
    # Update intervals
    dashboard_update_interval: float = 0.1  # 100ms = 10 FPS
    json_export_interval: float = 1.0       # 1 second
    
    # Integration settings
    use_existing_entropy: bool = True
    use_quantum_engine: bool = True
    use_ferris_wheel: bool = True
    
    # File paths
    json_export_path: str = "data/entropy_live.json"
    backup_export_path: str = "data/entropy_backup.json"

class EntropyBridge:
    """
    Main entropy bridge connecting quantum engine to Schwabot systems
    """
    
    def __init__(self, config: Optional[EntropyBridgeConfig] = None):
        self.config = config or EntropyBridgeConfig()
        self.log = logging.getLogger("entropy.bridge")
        
        # Data storage
        self.entropy_history: deque[EntropyFlowData] = deque(maxlen=self.config.history_size)
        self.current_data: Optional[EntropyFlowData] = None
        
        # Component integrations
        self.entropy_tracker = None
        self.quantum_engine = None
        self.hash_profit_matrix = None
        self.ferris_wheel = None
        
        # WebSocket server
        self.websocket_clients = set()
        self.websocket_server = None
        
        # Update tracking
        self.last_dashboard_update = 0.0
        self.last_json_export = 0.0
        self.frame_count = 0
        
        # Event callbacks
        self.data_update_callbacks: List[Callable] = []
        
        self._init_components()
        self.log.info("EntropyBridge initialized")
    
    def _init_components(self):
        """Initialize integrated components"""
        
        # Existing entropy tracker
        if self.config.use_existing_entropy and EntropyTracker:
            try:
                self.entropy_tracker = EntropyTracker()
                self.log.info("✓ EntropyTracker integration enabled")
            except Exception as e:
                self.log.warning(f"EntropyTracker init failed: {e}")
        
        # Quantum anti-pole engine
        if self.config.use_quantum_engine and QuantumAntiPoleEngine:
            try:
                qa_config = QAConfig(
                    use_gpu=True,
                    field_size=64,
                    debug_mode=False,
                    use_entropy_tracker=True,
                    use_thermal_manager=True,
                    use_ferris_wheel=True
                )
                self.quantum_engine = QuantumAntiPoleEngine(qa_config)
                self.log.info("✓ QuantumAntiPoleEngine integration enabled")
            except Exception as e:
                self.log.warning(f"QuantumAntiPoleEngine init failed: {e}")
        
        # Hash profit matrix
        if HashProfitMatrix:
            try:
                self.hash_profit_matrix = HashProfitMatrix()
                self.log.info("✓ HashProfitMatrix integration enabled")
            except Exception as e:
                self.log.warning(f"HashProfitMatrix init failed: {e}")
        
        # Ferris wheel scheduler
        if self.config.use_ferris_wheel and FerrisWheelScheduler:
            try:
                self.ferris_wheel = FerrisWheelScheduler()
                self.log.info("✓ FerrisWheelScheduler integration enabled")
            except Exception as e:
                self.log.warning(f"FerrisWheelScheduler init failed: {e}")
    
    async def process_market_tick(self, price: float, volume: float, 
                                timestamp: Optional[datetime] = None) -> EntropyFlowData:
        """
        Process market tick through complete entropy analysis pipeline
        """
        timestamp = timestamp or datetime.utcnow()
        start_time = time.perf_counter()
        
        # Gather entropy data from multiple sources
        entropy_data = await self._gather_entropy_data(price, volume, timestamp)
        
        # Process through quantum engine if available
        quantum_data = await self._process_quantum_analysis(price, volume, timestamp)
        
        # Integrate with ferris wheel strategy
        strategy_data = await self._process_strategy_integration(price, volume, timestamp)
        
        # Combine all data sources
        flow_data = self._combine_entropy_sources(
            price, volume, timestamp, entropy_data, quantum_data, strategy_data
        )
        
        # Calculate performance metrics
        computation_time = (time.perf_counter() - start_time) * 1000
        flow_data.computation_time_ms = computation_time
        
        # Store and distribute
        self._store_data(flow_data)
        await self._distribute_data(flow_data)
        
        self.frame_count += 1
        return flow_data
    
    async def _gather_entropy_data(self, price: float, volume: float, 
                                 timestamp: datetime) -> Dict[str, Any]:
        """Gather entropy data from existing tracker"""
        entropy_data = {
            'hash_entropy': 0.0,
            'entropy_tier': 'UNKNOWN',
            'hash_metrics': {}
        }
        
        if self.entropy_tracker:
            try:
                # Call existing entropy tracker
                result = self.entropy_tracker.calculate_entropy(price, volume)
                if isinstance(result, dict):
                    entropy_data['hash_entropy'] = result.get('entropy', 0.0)
                    entropy_data['entropy_tier'] = result.get('tier', 'BRONZE')
                    entropy_data['hash_metrics'] = result
                else:
                    entropy_data['hash_entropy'] = float(result) if result else 0.0
            except Exception as e:
                self.log.warning(f"EntropyTracker error: {e}")
        
        return entropy_data
    
    async def _process_quantum_analysis(self, price: float, volume: float, 
                                      timestamp: datetime) -> Dict[str, Any]:
        """Process through quantum anti-pole engine"""
        quantum_data = {
            'quantum_entropy': 0.0,
            'ap_rsi': 50.0,
            'coherence': 0.0,
            'pole_count': 0,
            'vector_strength': 0.0,
            'signal_strength': 0.0,
            'recommendation': 'HOLD',
            'confidence': 0.0
        }
        
        if self.quantum_engine:
            try:
                # Process through quantum engine
                frame = await self.quantum_engine.process_tick(price, volume, timestamp)
                
                # Extract key metrics
                quantum_data['quantum_entropy'] = frame.quantum_state.entropy
                quantum_data['ap_rsi'] = frame.ap_rsi
                quantum_data['coherence'] = frame.quantum_state.coherence
                quantum_data['pole_count'] = len(frame.complex_poles)
                quantum_data['vector_strength'] = float(frame.vector_field.magnitude.mean())
                
                # Get trading signals
                signals = self.quantum_engine.get_trading_signals(frame)
                quantum_data['signal_strength'] = signals.get('combined_signal', 0.0)
                quantum_data['recommendation'] = signals.get('recommendation', 'HOLD')
                quantum_data['confidence'] = signals.get('pole_stability_ratio', 0.0)
                
            except Exception as e:
                self.log.warning(f"QuantumEngine error: {e}")
        
        return quantum_data
    
    async def _process_strategy_integration(self, price: float, volume: float, 
                                          timestamp: datetime) -> Dict[str, Any]:
        """Integrate with ferris wheel strategy system"""
        strategy_data = {
            'current_strategy': 'NEUTRAL',
            'strategy_confidence': 0.0,
            'wheel_position': 0.0
        }
        
        if self.ferris_wheel:
            try:
                wheel_state = self.ferris_wheel.get_current_state()
                strategy_data['current_strategy'] = wheel_state.get('current_tier', 'NEUTRAL')
                strategy_data['strategy_confidence'] = wheel_state.get('confidence', 0.0)
                strategy_data['wheel_position'] = wheel_state.get('position', 0.0)
            except Exception as e:
                self.log.warning(f"FerrisWheel error: {e}")
        
        return strategy_data
    
    def _combine_entropy_sources(self, price: float, volume: float, timestamp: datetime,
                               entropy_data: Dict, quantum_data: Dict, 
                               strategy_data: Dict) -> EntropyFlowData:
        """Combine entropy data from all sources"""
        
        # Weighted combination of entropy sources
        hash_entropy = entropy_data.get('hash_entropy', 0.0)
        quantum_entropy = quantum_data.get('quantum_entropy', 0.0)
        
        # Combined entropy with weighting
        if hash_entropy > 0 and quantum_entropy > 0:
            combined_entropy = 0.6 * hash_entropy + 0.4 * quantum_entropy
        elif hash_entropy > 0:
            combined_entropy = hash_entropy
        elif quantum_entropy > 0:
            combined_entropy = quantum_entropy
        else:
            combined_entropy = 0.0
        
        # Determine entropy tier based on combined metrics
        entropy_tier = self._determine_entropy_tier(
            combined_entropy, 
            quantum_data.get('ap_rsi', 50.0),
            quantum_data.get('coherence', 0.0)
        )
        
        # Enhanced recommendation based on multiple signals
        recommendation = self._enhance_recommendation(
            quantum_data.get('recommendation', 'HOLD'),
            entropy_tier,
            strategy_data.get('current_strategy', 'NEUTRAL')
        )
        
        return EntropyFlowData(
            timestamp=timestamp,
            price=price,
            volume=volume,
            hash_entropy=hash_entropy,
            quantum_entropy=quantum_entropy,
            combined_entropy=combined_entropy,
            entropy_tier=entropy_tier,
            ap_rsi=quantum_data.get('ap_rsi', 50.0),
            coherence=quantum_data.get('coherence', 0.0),
            pole_count=quantum_data.get('pole_count', 0),
            vector_strength=quantum_data.get('vector_strength', 0.0),
            signal_strength=quantum_data.get('signal_strength', 0.0),
            recommendation=recommendation,
            confidence=quantum_data.get('confidence', 0.0),
            computation_time_ms=0.0  # Will be set by caller
        )
    
    def _determine_entropy_tier(self, combined_entropy: float, ap_rsi: float, 
                              coherence: float) -> str:
        """Determine entropy tier from combined metrics"""
        # Multi-factor tier determination
        score = 0.0
        
        # Entropy contribution (0-40 points)
        if combined_entropy > 0.8:
            score += 40
        elif combined_entropy > 0.6:
            score += 30
        elif combined_entropy > 0.4:
            score += 20
        elif combined_entropy > 0.2:
            score += 10
        
        # AP-RSI contribution (0-30 points)
        if ap_rsi > 70 or ap_rsi < 30:  # Extreme values
            score += 30
        elif ap_rsi > 60 or ap_rsi < 40:  # Moderate extremes
            score += 20
        else:
            score += 10
        
        # Coherence contribution (0-30 points)
        if coherence > 0.7:
            score += 30
        elif coherence > 0.5:
            score += 20
        elif coherence > 0.3:
            score += 10
        
        # Tier assignment
        if score >= 80:
            return "PLATINUM"
        elif score >= 60:
            return "GOLD"
        elif score >= 40:
            return "SILVER"
        elif score >= 20:
            return "BRONZE"
        else:
            return "NEUTRAL"
    
    def _enhance_recommendation(self, quantum_rec: str, entropy_tier: str, 
                              strategy: str) -> str:
        """Enhance recommendation using multiple factors"""
        # Convert to numerical scores for combination
        rec_scores = {
            'STRONG_SELL': -2, 'SELL': -1, 'HOLD': 0, 'BUY': 1, 'STRONG_BUY': 2
        }
        
        tier_scores = {
            'PLATINUM': 2, 'GOLD': 1, 'SILVER': 0, 'BRONZE': -1, 'NEUTRAL': 0
        }
        
        strategy_scores = {
            'AGGRESSIVE': 1, 'MODERATE': 0, 'CONSERVATIVE': -1, 'NEUTRAL': 0
        }
        
        # Weighted combination
        quantum_score = rec_scores.get(quantum_rec.upper(), 0)
        tier_score = tier_scores.get(entropy_tier, 0)
        strategy_score = strategy_scores.get(strategy.upper(), 0)
        
        combined_score = 0.5 * quantum_score + 0.3 * tier_score + 0.2 * strategy_score
        
        # Convert back to recommendation
        if combined_score >= 1.5:
            return "STRONG_BUY"
        elif combined_score >= 0.5:
            return "BUY"
        elif combined_score <= -1.5:
            return "STRONG_SELL"
        elif combined_score <= -0.5:
            return "SELL"
        else:
            return "HOLD"
    
    def _store_data(self, flow_data: EntropyFlowData):
        """Store data in history buffer"""
        self.entropy_history.append(flow_data)
        self.current_data = flow_data
    
    async def _distribute_data(self, flow_data: EntropyFlowData):
        """Distribute data to various consumers"""
        current_time = time.time()
        
        # Dashboard updates (high frequency)
        if current_time - self.last_dashboard_update >= self.config.dashboard_update_interval:
            await self._update_dashboard(flow_data)
            self.last_dashboard_update = current_time
        
        # JSON export (lower frequency)
        if current_time - self.last_json_export >= self.config.json_export_interval:
            await self._export_json(flow_data)
            self.last_json_export = current_time
        
        # Notify callbacks
        for callback in self.data_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(flow_data)
                else:
                    callback(flow_data)
            except Exception as e:
                self.log.warning(f"Callback error: {e}")
    
    async def _update_dashboard(self, flow_data: EntropyFlowData):
        """Update dashboard via WebSocket"""
        if not self.websocket_clients:
            return
        
        # Create dashboard update message
        dashboard_data = {
            'type': 'entropy_update',
            'timestamp': flow_data.timestamp.isoformat(),
            'data': {
                'price': flow_data.price,
                'volume': flow_data.volume,
                'hash_entropy': flow_data.hash_entropy,
                'quantum_entropy': flow_data.quantum_entropy,
                'combined_entropy': flow_data.combined_entropy,
                'entropy_tier': flow_data.entropy_tier,
                'ap_rsi': flow_data.ap_rsi,
                'coherence': flow_data.coherence,
                'pole_count': flow_data.pole_count,
                'vector_strength': flow_data.vector_strength,
                'signal_strength': flow_data.signal_strength,
                'recommendation': flow_data.recommendation,
                'confidence': flow_data.confidence
            },
            'performance': {
                'computation_time_ms': flow_data.computation_time_ms,
                'frame_count': self.frame_count
            }
        }
        
        message = json.dumps(dashboard_data)
        
        # Broadcast to all connected clients
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                self.log.warning(f"WebSocket send error: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def _export_json(self, flow_data: EntropyFlowData):
        """Export current state to JSON file"""
        try:
            # Create export data
            export_data = {
                'timestamp': flow_data.timestamp.isoformat(),
                'current': asdict(flow_data),
                'history': [asdict(item) for item in list(self.entropy_history)[-100:]],  # Last 100 items
                'statistics': self.get_statistics(),
                'metadata': {
                    'frame_count': self.frame_count,
                    'history_size': len(self.entropy_history),
                    'components_active': {
                        'entropy_tracker': self.entropy_tracker is not None,
                        'quantum_engine': self.quantum_engine is not None,
                        'ferris_wheel': self.ferris_wheel is not None
                    }
                }
            }
            
            # Write to main file
            with open(self.config.json_export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Backup to secondary file
            with open(self.config.backup_export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        except Exception as e:
            self.log.error(f"JSON export error: {e}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for dashboard connections"""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                self.config.websocket_port
            )
            self.log.info(f"🌐 Entropy bridge WebSocket server started on port {self.config.websocket_port}")
        except Exception as e:
            self.log.error(f"WebSocket server start error: {e}")
    
    async def stop_websocket_server(self):
        """Stop WebSocket server"""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            self.log.info("WebSocket server stopped")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        client_addr = websocket.remote_address
        self.websocket_clients.add(websocket)
        self.log.info(f"Dashboard client connected: {client_addr}")
        
        try:
            # Send initial data if available
            if self.current_data:
                await self._update_dashboard(self.current_data)
            
            # Keep connection alive
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            self.log.info(f"Dashboard client disconnected: {client_addr}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback for data updates"""
        self.data_update_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable):
        """Remove data update callback"""
        if callback in self.data_update_callbacks:
            self.data_update_callbacks.remove(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        if not self.entropy_history:
            return {}
        
        # Calculate statistics from history
        history_list = list(self.entropy_history)
        recent_data = history_list[-100:]  # Last 100 frames
        
        return {
            'total_frames': len(history_list),
            'avg_computation_time': sum(d.computation_time_ms for d in recent_data) / len(recent_data),
            'avg_entropy': sum(d.combined_entropy for d in recent_data) / len(recent_data),
            'avg_coherence': sum(d.coherence for d in recent_data) / len(recent_data),
            'tier_distribution': {
                tier: sum(1 for d in recent_data if d.entropy_tier == tier)
                for tier in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'NEUTRAL']
            },
            'recommendation_distribution': {
                rec: sum(1 for d in recent_data if d.recommendation == rec)
                for rec in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
            },
            'connected_clients': len(self.websocket_clients)
        }
    
    def get_current_state(self) -> Optional[EntropyFlowData]:
        """Get current entropy state"""
        return self.current_data
    
    def get_history(self, limit: Optional[int] = None) -> List[EntropyFlowData]:
        """Get entropy history"""
        history = list(self.entropy_history)
        if limit:
            return history[-limit:]
        return history

# Example integration function for existing systems
async def integrate_with_existing_systems():
    """Example of how to integrate entropy bridge with existing Schwabot systems"""
    
    # Create and configure bridge
    config = EntropyBridgeConfig(
        websocket_port=8767,
        json_export_path="data/live_entropy.json",
        use_existing_entropy=True,
        use_quantum_engine=True
    )
    
    bridge = EntropyBridge(config)
    
    # Start WebSocket server for dashboard
    await bridge.start_websocket_server()
    
    # Example callback for external systems
    def on_entropy_update(flow_data: EntropyFlowData):
        if flow_data.entropy_tier in ['PLATINUM', 'GOLD']:
            print(f"🚨 High entropy detected: {flow_data.entropy_tier} at ${flow_data.price:,.2f}")
    
    bridge.add_data_callback(on_entropy_update)
    
    return bridge

if __name__ == "__main__":
    import asyncio
    
    async def test_entropy_bridge():
        """Test the entropy bridge"""
        logging.basicConfig(level=logging.INFO)
        
        bridge = await integrate_with_existing_systems()
        
        print("🌉 Testing Entropy Bridge...")
        
        # Simulate market data
        base_price = 45000.0
        base_volume = 1000000.0
        
        for i in range(20):
            import math
            import numpy as np
            
            price = base_price + 1000 * math.sin(i * 0.2) + np.random.randn() * 300
            volume = base_volume + np.random.randn() * 150000
            
            # Process through bridge
            flow_data = await bridge.process_market_tick(price, volume)
            
            if i % 5 == 0:
                print(f"\n📊 Tick {i}: ${price:,.2f}")
                print(f"   🔗 Hash Entropy: {flow_data.hash_entropy:.3f}")
                print(f"   ⚛️  Quantum Entropy: {flow_data.quantum_entropy:.3f}")
                print(f"   🎯 Combined: {flow_data.combined_entropy:.3f} ({flow_data.entropy_tier})")
                print(f"   📈 AP-RSI: {flow_data.ap_rsi:.1f}")
                print(f"   🔮 Recommendation: {flow_data.recommendation}")
                print(f"   ⚡ Time: {flow_data.computation_time_ms:.1f}ms")
            
            await asyncio.sleep(0.1)  # 10 FPS
        
        # Show statistics
        stats = bridge.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   • Total frames: {stats.get('total_frames', 0)}")
        print(f"   • Avg computation time: {stats.get('avg_computation_time', 0):.1f}ms")
        print(f"   • Tier distribution: {stats.get('tier_distribution', {})}")
        
        await bridge.stop_websocket_server()
    
    asyncio.run(test_entropy_bridge()) 