"""
Dashboard Integration v1.0
==========================

Advanced integration layer connecting Schwabot backend systems
to the React TSX dashboard with real-time data streaming.

Provides:
- WebSocket API for React dashboard
- REST API endpoints
- Real-time entropy and anti-pole data streaming
- Trading signal distribution
- Performance monitoring
"""

import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set
from aiohttp import web, WSMsgType
from aiohttp.web_ws import WebSocketResponse
import aiohttp_cors
from collections import deque
import numpy as np

# Core Schwabot imports
try:
    from .entropy_bridge import EntropyBridge, EntropyFlowData, EntropyBridgeConfig
    from .quantum_antipole_engine import QuantumAntiPoleEngine, QAConfig
    from .profit_navigator import AntiPoleProfitNavigator
    from .ferris_wheel_scheduler import FerrisWheelScheduler
    from .thermal_zone_manager import ThermalZoneManager
except ImportError:
    EntropyBridge = None
    EntropyFlowData = None
    QuantumAntiPoleEngine = None
    AntiPoleProfitNavigator = None
    FerrisWheelScheduler = None
    ThermalZoneManager = None

@dataclass
class DashboardConfig:
    """Configuration for dashboard integration"""
    # Server settings
    host: str = "localhost"
    port: int = 8768
    
    # Data streaming
    max_history_points: int = 1000
    update_frequency: float = 10.0  # Hz
    
    # CORS settings
    cors_origins: List[str] = None
    
    # Component integration
    enable_entropy_bridge: bool = True
    enable_profit_navigator: bool = True
    enable_thermal_monitoring: bool = True
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:3001"]

@dataclass
class DashboardData:
    """Complete dashboard data package"""
    timestamp: datetime
    market_data: Dict[str, float]
    entropy_data: Dict[str, Any]
    antipole_data: Dict[str, Any]
    trading_signals: Dict[str, Any]
    thermal_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    system_health: Dict[str, Any]

class DashboardIntegration:
    """
    Main dashboard integration server providing real-time data to React frontend
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.log = logging.getLogger("dashboard.integration")
        
        # Web application
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocketResponse] = set()
        
        # Data storage
        self.current_data: Optional[DashboardData] = None
        self.data_history: deque[DashboardData] = deque(maxlen=self.config.max_history_points)
        
        # Component integrations
        self.entropy_bridge: Optional[EntropyBridge] = None
        self.profit_navigator: Optional[AntiPoleProfitNavigator] = None
        self.thermal_manager: Optional[ThermalZoneManager] = None
        
        # Update tracking
        self.last_update = 0.0
        self.update_interval = 1.0 / self.config.update_frequency
        self.frame_count = 0
        
        # Initialize components
        self._init_components()
        
        self.log.info(f"DashboardIntegration initialized on {self.config.host}:{self.config.port}")
    
    def _init_components(self):
        """Initialize integrated components"""
        
        # Entropy bridge
        if self.config.enable_entropy_bridge and EntropyBridge:
            try:
                entropy_config = EntropyBridgeConfig(
                    websocket_port=8767,  # Different port to avoid conflicts
                    use_existing_entropy=True,
                    use_quantum_engine=True
                )
                self.entropy_bridge = EntropyBridge(entropy_config)
                self.log.info("✓ EntropyBridge integration enabled")
            except Exception as e:
                self.log.warning(f"EntropyBridge init failed: {e}")
        
        # Profit navigator
        if self.config.enable_profit_navigator and AntiPoleProfitNavigator:
            try:
                self.profit_navigator = AntiPoleProfitNavigator(
                    initial_balance=100000.0,
                    max_position_size=0.25
                )
                self.log.info("✓ AntiPoleProfitNavigator integration enabled")
            except Exception as e:
                self.log.warning(f"AntiPoleProfitNavigator init failed: {e}")
        
        # Thermal monitoring
        if self.config.enable_thermal_monitoring and ThermalZoneManager:
            try:
                self.thermal_manager = ThermalZoneManager()
                self.log.info("✓ ThermalZoneManager integration enabled")
            except Exception as e:
                self.log.warning(f"ThermalZoneManager init failed: {e}")
    
    def setup_routes(self):
        """Setup HTTP routes and WebSocket endpoints"""
        
        # REST API endpoints
        self.app.router.add_get('/api/status', self.get_status)
        self.app.router.add_get('/api/current-data', self.get_current_data)
        self.app.router.add_get('/api/history/{limit}', self.get_history)
        self.app.router.add_get('/api/statistics', self.get_statistics)
        self.app.router.add_get('/api/health', self.get_health)
        
        # Configuration endpoints
        self.app.router.add_get('/api/config', self.get_config)
        self.app.router.add_post('/api/config', self.set_config)
        
        # WebSocket endpoint for real-time data
        self.app.router.add_get('/ws/live-data', self.websocket_handler)
        
        # Static file serving (for development)
        self.app.router.add_static('/', 'static/', name='static')
    
    def setup_cors(self):
        """Setup CORS for cross-origin requests"""
        cors = aiohttp_cors.setup(self.app, defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            ) for origin in self.config.cors_origins
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def process_market_tick(self, price: float, volume: float, 
                                timestamp: Optional[datetime] = None) -> DashboardData:
        """
        Process market tick and generate complete dashboard data
        """
        timestamp = timestamp or datetime.utcnow()
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update < self.update_interval:
            return self.current_data
        
        # Gather data from all sources
        dashboard_data = await self._gather_dashboard_data(price, volume, timestamp)
        
        # Store and distribute
        self.current_data = dashboard_data
        self.data_history.append(dashboard_data)
        
        # Broadcast to WebSocket connections
        await self._broadcast_to_websockets(dashboard_data)
        
        self.last_update = current_time
        self.frame_count += 1
        
        return dashboard_data
    
    async def _gather_dashboard_data(self, price: float, volume: float, 
                                   timestamp: datetime) -> DashboardData:
        """Gather data from all integrated components"""
        
        # Market data
        market_data = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp.timestamp()
        }
        
        # Entropy data
        entropy_data = await self._gather_entropy_data(price, volume, timestamp)
        
        # Anti-pole data
        antipole_data = await self._gather_antipole_data(price, volume, timestamp)
        
        # Trading signals
        trading_signals = await self._gather_trading_signals(price, volume, timestamp)
        
        # Thermal data
        thermal_data = await self._gather_thermal_data()
        
        # Performance metrics
        performance_metrics = self._gather_performance_metrics()
        
        # System health
        system_health = self._gather_system_health()
        
        return DashboardData(
            timestamp=timestamp,
            market_data=market_data,
            entropy_data=entropy_data,
            antipole_data=antipole_data,
            trading_signals=trading_signals,
            thermal_data=thermal_data,
            performance_metrics=performance_metrics,
            system_health=system_health
        )
    
    async def _gather_entropy_data(self, price: float, volume: float, 
                                 timestamp: datetime) -> Dict[str, Any]:
        """Gather entropy data from bridge"""
        entropy_data = {
            'hash_entropy': 0.0,
            'quantum_entropy': 0.0,
            'combined_entropy': 0.0,
            'entropy_tier': 'NEUTRAL',
            'coherence': 0.0,
            'ap_rsi': 50.0
        }
        
        if self.entropy_bridge:
            try:
                flow_data = await self.entropy_bridge.process_market_tick(price, volume, timestamp)
                entropy_data.update({
                    'hash_entropy': flow_data.hash_entropy,
                    'quantum_entropy': flow_data.quantum_entropy,
                    'combined_entropy': flow_data.combined_entropy,
                    'entropy_tier': flow_data.entropy_tier,
                    'coherence': flow_data.coherence,
                    'ap_rsi': flow_data.ap_rsi,
                    'computation_time': flow_data.computation_time_ms
                })
            except Exception as e:
                self.log.warning(f"Entropy data gathering error: {e}")
        
        return entropy_data
    
    async def _gather_antipole_data(self, price: float, volume: float, 
                                  timestamp: datetime) -> Dict[str, Any]:
        """Gather anti-pole specific data"""
        antipole_data = {
            'pole_count': 0,
            'vector_strength': 0.0,
            'capt_transform': [],
            'phi_surface': [],
            'stability_ratio': 0.0
        }
        
        if self.entropy_bridge and self.entropy_bridge.quantum_engine:
            try:
                # Get last processed frame from quantum engine
                if hasattr(self.entropy_bridge.quantum_engine, 'current_quantum_state'):
                    # Extract anti-pole specific metrics
                    engine = self.entropy_bridge.quantum_engine
                    if hasattr(engine, 'frame_count') and engine.frame_count > 0:
                        # Get performance metrics
                        metrics = engine.get_performance_metrics()
                        antipole_data.update({
                            'frames_processed': metrics.get('frames_processed', 0),
                            'buffer_fill': metrics.get('buffer_fill', 0.0)
                        })
            except Exception as e:
                self.log.warning(f"Anti-pole data gathering error: {e}")
        
        return antipole_data
    
    async def _gather_trading_signals(self, price: float, volume: float, 
                                    timestamp: datetime) -> Dict[str, Any]:
        """Gather trading signals from profit navigator"""
        trading_signals = {
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'signal_strength': 0.0,
            'opportunities': [],
            'portfolio_value': 0.0,
            'unrealized_pnl': 0.0
        }
        
        if self.profit_navigator:
            try:
                # Process through profit navigator
                tick_report = await self.profit_navigator.process_market_tick(price, volume, timestamp)
                
                trading_signals.update({
                    'opportunities': [
                        {
                            'type': opp.get('opportunity_type', 'UNKNOWN'),
                            'confidence': opp.get('confidence', 0.0),
                            'tier': opp.get('profit_tier', 'BRONZE'),
                            'expected_return': opp.get('expected_return', 0.0)
                        }
                        for opp in tick_report.get('opportunities', [])
                    ],
                    'portfolio_value': tick_report.get('portfolio_state', {}).get('total_value', 0.0),
                    'unrealized_pnl': tick_report.get('portfolio_state', {}).get('unrealized_pnl', 0.0),
                    'recommendations': tick_report.get('recommendations', [])
                })
                
                # Extract primary recommendation
                recommendations = tick_report.get('recommendations', [])
                if recommendations:
                    primary_rec = recommendations[0]
                    trading_signals['recommendation'] = primary_rec.get('action', 'HOLD')
                    trading_signals['confidence'] = primary_rec.get('confidence', 0.0)
                
            except Exception as e:
                self.log.warning(f"Trading signals gathering error: {e}")
        
        return trading_signals
    
    async def _gather_thermal_data(self) -> Dict[str, Any]:
        """Gather thermal monitoring data"""
        thermal_data = {
            'temperature': 25.0,
            'thermal_state': 'NORMAL',
            'cooldown_active': False,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'safe_to_trade': True
        }
        
        if self.thermal_manager:
            try:
                thermal_state = self.thermal_manager.get_thermal_state()
                thermal_data.update({
                    'temperature': thermal_state.get('temperature', 25.0),
                    'thermal_state': thermal_state.get('state', 'NORMAL'),
                    'cooldown_active': thermal_state.get('cooldown_active', False),
                    'cpu_usage': thermal_state.get('cpu_usage', 0.0),
                    'memory_usage': thermal_state.get('memory_usage', 0.0),
                    'safe_to_trade': thermal_state.get('safe', True)
                })
            except Exception as e:
                self.log.warning(f"Thermal data gathering error: {e}")
        
        return thermal_data
    
    def _gather_performance_metrics(self) -> Dict[str, Any]:
        """Gather performance metrics"""
        return {
            'frame_count': self.frame_count,
            'update_frequency': self.config.update_frequency,
            'websocket_connections': len(self.websocket_connections),
            'data_history_size': len(self.data_history),
            'last_update': self.last_update
        }
    
    def _gather_system_health(self) -> Dict[str, Any]:
        """Gather system health status"""
        return {
            'entropy_bridge_active': self.entropy_bridge is not None,
            'profit_navigator_active': self.profit_navigator is not None,
            'thermal_manager_active': self.thermal_manager is not None,
            'websocket_server_running': True,
            'api_server_running': True
        }
    
    async def _broadcast_to_websockets(self, dashboard_data: DashboardData):
        """Broadcast data to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        # Create message
        message = {
            'type': 'dashboard_update',
            'timestamp': dashboard_data.timestamp.isoformat(),
            'data': {
                'market': dashboard_data.market_data,
                'entropy': dashboard_data.entropy_data,
                'antipole': dashboard_data.antipole_data,
                'trading': dashboard_data.trading_signals,
                'thermal': dashboard_data.thermal_data,
                'performance': dashboard_data.performance_metrics,
                'health': dashboard_data.system_health
            }
        }
        
        message_json = json.dumps(message, default=str)
        
        # Send to all connections
        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_str(message_json)
            except Exception as e:
                self.log.warning(f"WebSocket send error: {e}")
                disconnected.add(ws)
        
        # Remove disconnected connections
        self.websocket_connections -= disconnected
    
    # HTTP Route Handlers
    async def get_status(self, request):
        """Get system status"""
        return web.json_response({
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'frame_count': self.frame_count,
            'websocket_connections': len(self.websocket_connections),
            'components': {
                'entropy_bridge': self.entropy_bridge is not None,
                'profit_navigator': self.profit_navigator is not None,
                'thermal_manager': self.thermal_manager is not None
            }
        })
    
    async def get_current_data(self, request):
        """Get current dashboard data"""
        if not self.current_data:
            return web.json_response({'error': 'No data available'}, status=404)
        
        return web.json_response(asdict(self.current_data), default=str)
    
    async def get_history(self, request):
        """Get historical data"""
        limit = int(request.match_info.get('limit', 100))
        limit = min(limit, len(self.data_history))
        
        history = list(self.data_history)[-limit:]
        return web.json_response([asdict(item) for item in history], default=str)
    
    async def get_statistics(self, request):
        """Get system statistics"""
        stats = {
            'total_frames': self.frame_count,
            'update_frequency': self.config.update_frequency,
            'history_size': len(self.data_history),
            'websocket_connections': len(self.websocket_connections)
        }
        
        # Add component statistics
        if self.entropy_bridge:
            stats['entropy_bridge'] = self.entropy_bridge.get_statistics()
        
        if self.profit_navigator:
            stats['profit_navigator'] = self.profit_navigator.get_comprehensive_status()
        
        return web.json_response(stats, default=str)
    
    async def get_health(self, request):
        """Get system health check"""
        health = {
            'healthy': True,
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        # Check component health
        try:
            if self.entropy_bridge:
                health['components']['entropy_bridge'] = 'healthy'
            else:
                health['components']['entropy_bridge'] = 'disabled'
            
            if self.profit_navigator:
                health['components']['profit_navigator'] = 'healthy'
            else:
                health['components']['profit_navigator'] = 'disabled'
            
            if self.thermal_manager:
                thermal_state = self.thermal_manager.get_thermal_state()
                health['components']['thermal_manager'] = 'healthy' if thermal_state.get('safe', True) else 'warning'
            else:
                health['components']['thermal_manager'] = 'disabled'
                
        except Exception as e:
            health['healthy'] = False
            health['error'] = str(e)
        
        return web.json_response(health)
    
    async def get_config(self, request):
        """Get current configuration"""
        return web.json_response(asdict(self.config))
    
    async def set_config(self, request):
        """Update configuration"""
        try:
            data = await request.json()
            # Update configuration fields
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            return web.json_response({'status': 'updated'})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        client_addr = request.remote
        self.log.info(f"WebSocket client connected: {client_addr}")
        
        try:
            # Send initial data if available
            if self.current_data:
                await self._broadcast_to_websockets(self.current_data)
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    self.log.error(f'WebSocket error: {ws.exception()}')
                    break
        
        except Exception as e:
            self.log.warning(f"WebSocket error: {e}")
        
        finally:
            self.websocket_connections.discard(ws)
            self.log.info(f"WebSocket client disconnected: {client_addr}")
        
        return ws
    
    async def _handle_websocket_message(self, ws: WebSocketResponse, data: Dict):
        """Handle incoming WebSocket message"""
        msg_type = data.get('type')
        
        if msg_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong', 'timestamp': time.time()}))
        
        elif msg_type == 'get_history':
            limit = data.get('limit', 100)
            history = list(self.data_history)[-limit:]
            await ws.send_str(json.dumps({
                'type': 'history_data',
                'data': [asdict(item) for item in history]
            }, default=str))
        
        elif msg_type == 'get_statistics':
            stats = await self.get_statistics(None)
            await ws.send_str(json.dumps({
                'type': 'statistics_data',
                'data': json.loads(stats.text)
            }))
    
    async def start_server(self):
        """Start the dashboard integration server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        self.log.info(f"🚀 Dashboard integration server started on http://{self.config.host}:{self.config.port}")
        self.log.info(f"📊 WebSocket endpoint: ws://{self.config.host}:{self.config.port}/ws/live-data")
        self.log.info(f"🔌 API base URL: http://{self.config.host}:{self.config.port}/api/")
    
    async def stop_server(self):
        """Stop the dashboard integration server"""
        # Close all WebSocket connections
        for ws in list(self.websocket_connections):
            await ws.close()
        
        self.log.info("Dashboard integration server stopped")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_dashboard_integration():
        """Test the dashboard integration"""
        logging.basicConfig(level=logging.INFO)
        
        # Create dashboard integration
        config = DashboardConfig(
            host="localhost",
            port=8768,
            update_frequency=10.0,
            enable_entropy_bridge=True,
            enable_profit_navigator=True
        )
        
        dashboard = DashboardIntegration(config)
        
        # Start server
        await dashboard.start_server()
        
        print("🌐 Dashboard Integration Server running...")
        print(f"📊 Dashboard: http://localhost:8768/")
        print(f"🔌 WebSocket: ws://localhost:8768/ws/live-data")
        print(f"📡 API: http://localhost:8768/api/status")
        
        # Simulate market data for testing
        base_price = 45000.0
        base_volume = 1000000.0
        
        try:
            for i in range(100):
                import math
                import numpy as np
                
                price = base_price + 2000 * math.sin(i * 0.1) + np.random.randn() * 400
                volume = base_volume + np.random.randn() * 180000
                
                # Process market tick
                dashboard_data = await dashboard.process_market_tick(price, volume)
                
                if i % 10 == 0:
                    print(f"📈 Tick {i}: ${price:,.2f} | "
                          f"Entropy: {dashboard_data.entropy_data.get('combined_entropy', 0):.3f} | "
                          f"AP-RSI: {dashboard_data.entropy_data.get('ap_rsi', 50):.1f} | "
                          f"Connections: {len(dashboard.websocket_connections)}")
                
                await asyncio.sleep(1.0 / config.update_frequency)  # Maintain update frequency
        
        except KeyboardInterrupt:
            print("🛑 Stopping dashboard integration...")
        
        finally:
            await dashboard.stop_server()
    
    # Run test
    asyncio.run(test_dashboard_integration()) 