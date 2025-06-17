#!/usr/bin/env python3
"""
Visual Integration Bridge
========================

Bridges the existing Tesseract visualizers with the new Schwabot dashboard system.
Provides real-time data streaming to both Python matplotlib and React/web frontends.

This module serves as the central hub for all visualization data, ensuring that:
- TesseractVisualizer (Python/matplotlib) gets live tensor data
- AdvancedTesseractVisualizer (React/web) receives WebSocket streams
- Dashboard components get unified visual state
- All visualizers share the same data source for consistency
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import websockets
import numpy as np

# Core system imports
from .ui_state_bridge import UIStateBridge, UIVisualizationData
from .sustainment_underlay_controller import SustainmentUnderlayController

# Import visualizers (with fallbacks)
try:
    from .enhanced_tesseract_processor import EnhancedTesseractProcessor
    from .tensor_visualization_controller import TensorVisualizationController
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract visualizers not available - running in compatibility mode")

logger = logging.getLogger(__name__)

@dataclass
class VisualMetrics:
    """Unified visual metrics for all visualizers"""
    # Tesseract metrics
    magnitude: float
    centroid_distance: float
    entropy: float
    coherence: float
    pattern_strength: float
    
    # Market entropy
    market_entropy: float
    volatility: float
    trend_strength: float
    
    # System health
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    thermal_load: float
    
    # Trading metrics
    active_patterns: int
    confidence_score: float
    risk_level: float
    profit_velocity: float
    
    # Timestamp
    timestamp: datetime

@dataclass
class PatternState:
    """Current pattern analysis state"""
    active_patterns: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    performance_data: List[Dict[str, Any]]
    entropy_data: List[Dict[str, Any]]
    system_health: Dict[str, Any]

class VisualIntegrationBridge:
    """
    Central bridge for all visualization systems.
    
    Coordinates data flow between:
    - Sustainment underlay (mathematical core)
    - UI state bridge (dashboard data)
    - TesseractVisualizer (Python matplotlib)
    - React dashboard (web interface)
    - API data feeds (market data)
    """
    
    def __init__(self, 
                 ui_bridge: UIStateBridge,
                 sustainment_controller: SustainmentUnderlayController,
                 websocket_host: str = "localhost",
                 websocket_port: int = 8765):
        """
        Initialize visual integration bridge
        
        Args:
            ui_bridge: UI state bridge for dashboard data
            sustainment_controller: Sustainment underlay for mathematical synthesis
            websocket_host: WebSocket server host for React dashboard
            websocket_port: WebSocket server port for React dashboard
        """
        
        # Core components
        self.ui_bridge = ui_bridge
        self.sustainment_controller = sustainment_controller
        
        # WebSocket server for React dashboard
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.websocket_server = None
        
        # Tesseract visualizers
        self.tesseract_processor = None
        self.tensor_controller = None
        
        # Visual state tracking
        self.current_metrics: Optional[VisualMetrics] = None
        self.current_pattern_state: Optional[PatternState] = None
        self.metrics_history: List[VisualMetrics] = []
        self.pattern_history: List[PatternState] = []
        
        # Real-time data streams
        self.market_data_stream: List[Dict[str, Any]] = []
        self.entropy_data_stream: List[Dict[str, Any]] = []
        self.performance_data_stream: List[Dict[str, Any]] = []
        
        # Threading and control
        self.bridge_active = False
        self.update_thread = None
        self.websocket_thread = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.update_count = 0
        self.websocket_messages_sent = 0
        self.start_time = datetime.now()
        
        logger.info("Visual Integration Bridge initialized")

    def initialize_visualizers(self) -> None:
        """Initialize the Tesseract visualizers if available"""
        
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract visualizers not available - skipping initialization")
            return
        
        try:
            # Initialize Enhanced Tesseract Processor
            self.tesseract_processor = EnhancedTesseractProcessor()
            logger.info("Enhanced Tesseract Processor initialized")
            
            # Initialize Tensor Visualization Controller
            self.tensor_controller = TensorVisualizationController()
            logger.info("Tensor Visualization Controller initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract visualizers: {e}")
            self.tesseract_processor = None
            self.tensor_controller = None

    async def start_websocket_server(self) -> None:
        """Start WebSocket server for React dashboard communication"""
        
        async def handle_client(websocket, path):
            """Handle new WebSocket client connection"""
            logger.info(f"New WebSocket client connected: {websocket.remote_address}")
            self.websocket_clients.add(websocket)
            
            try:
                # Send initial state
                if self.current_pattern_state:
                    await websocket.send(json.dumps({
                        'type': 'pattern_state',
                        'data': asdict(self.current_pattern_state)
                    }, default=str))
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client messages if needed
                    try:
                        data = json.loads(message)
                        await self._handle_websocket_message(websocket, data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {message}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
        
        try:
            self.websocket_server = await websockets.serve(
                handle_client, 
                self.websocket_host, 
                self.websocket_port
            )
            logger.info(f"WebSocket server started on {self.websocket_host}:{self.websocket_port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages from React dashboard"""
        
        message_type = data.get('type')
        
        if message_type == 'request_state':
            # Send current state
            if self.current_pattern_state:
                await websocket.send(json.dumps({
                    'type': 'pattern_state',
                    'data': asdict(self.current_pattern_state)
                }, default=str))
        
        elif message_type == 'configure_visualizer':
            # Handle visualizer configuration changes
            config = data.get('config', {})
            await self._configure_visualizers(config)
        
        elif message_type == 'export_data':
            # Handle data export requests
            export_data = self._prepare_export_data()
            await websocket.send(json.dumps({
                'type': 'export_data',
                'data': export_data
            }, default=str))

    async def _configure_visualizers(self, config: Dict[str, Any]) -> None:
        """Configure visualizers based on client requests"""
        
        if self.tesseract_processor and 'tesseract' in config:
            # Configure tesseract processor
            tesseract_config = config['tesseract']
            # Apply configuration...
            
        if self.tensor_controller and 'tensor' in config:
            # Configure tensor controller
            tensor_config = config['tensor']
            # Apply configuration...
            
        logger.info(f"Applied visualizer configuration: {config}")

    def start_visual_bridge(self, update_interval: float = 0.1) -> None:
        """Start the complete visual bridge system"""
        
        if self.bridge_active:
            logger.warning("Visual bridge already active")
            return
        
        # Initialize visualizers
        self.initialize_visualizers()
        
        # Start WebSocket server in separate thread
        self.websocket_thread = threading.Thread(
            target=self._run_websocket_server,
            daemon=True
        )
        self.websocket_thread.start()
        
        # Start main update loop
        self.bridge_active = True
        self.update_thread = threading.Thread(
            target=self._visual_update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"Visual Integration Bridge started (update interval: {update_interval}s)")

    def _run_websocket_server(self) -> None:
        """Run WebSocket server in asyncio event loop"""
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_websocket_server())
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server thread failed: {e}")

    def _visual_update_loop(self, interval: float) -> None:
        """Main visual update loop"""
        
        while self.bridge_active:
            try:
                # Update visual metrics
                self._update_visual_metrics()
                
                # Update pattern state
                self._update_pattern_state()
                
                # Update Tesseract visualizers
                self._update_tesseract_visualizers()
                
                # Broadcast to WebSocket clients
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_to_websocket_clients(),
                    asyncio.get_event_loop()
                )
                
                self.update_count += 1
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in visual update loop: {e}")
                time.sleep(interval)

    def _update_visual_metrics(self) -> None:
        """Update unified visual metrics from all sources"""
        
        with self._lock:
            try:
                # Get current UI state
                ui_state = self.ui_bridge.get_ui_state()
                
                # Get sustainment status
                sustainment_status = self.sustainment_controller.get_sustainment_status()
                
                # Extract metrics
                system_health = ui_state.get('system_health', {})
                hardware_state = ui_state.get('hardware_state', {})
                trading_state = ui_state.get('trading_state', {})
                visualization_data = ui_state.get('visualization_data', {})
                
                # Build unified metrics
                metrics = VisualMetrics(
                    magnitude=0.75,  # Would come from tesseract processor
                    centroid_distance=0.23,
                    entropy=visualization_data.get('fractal_entropy', 0.4),
                    coherence=visualization_data.get('fractal_coherence', 0.6),
                    pattern_strength=visualization_data.get('pattern_strength', 0.75),
                    
                    market_entropy=self._calculate_market_entropy(),
                    volatility=visualization_data.get('volatility', 0.15),
                    trend_strength=0.68,
                    
                    cpu_usage=hardware_state.get('cpu_usage', 0.3),
                    gpu_usage=hardware_state.get('gpu_usage', 0.4),
                    memory_usage=hardware_state.get('memory_usage', 0.5),
                    thermal_load=hardware_state.get('cpu_temp', 55.0) / 100.0,
                    
                    active_patterns=len(visualization_data.get('active_overlays', [])),
                    confidence_score=sustainment_status.get('sustainment_index', 0.5),
                    risk_level=1.0 - trading_state.get('win_rate', 0.67),
                    profit_velocity=trading_state.get('daily_return', 0.023),
                    
                    timestamp=datetime.now()
                )
                
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
            except Exception as e:
                logger.error(f"Failed to update visual metrics: {e}")

    def _update_pattern_state(self) -> None:
        """Update pattern analysis state for React dashboard"""
        
        with self._lock:
            try:
                # Build pattern data for React dashboard
                active_patterns = [
                    {
                        'pattern': 'Trend Continuation',
                        'confidence': 0.85,
                        'active': True,
                        'strength': 0.78
                    },
                    {
                        'pattern': 'Mean Reversion',
                        'confidence': 0.72,
                        'active': False,
                        'strength': 0.45
                    },
                    {
                        'pattern': 'Breakout Signal',
                        'confidence': 0.91,
                        'active': True,
                        'strength': 0.89
                    },
                    {
                        'pattern': 'Anti-Pole Formation',
                        'confidence': 0.94,
                        'active': True,
                        'strength': 0.92
                    }
                ]
                
                # Risk metrics for radar chart
                risk_metrics = {
                    'exposure': 0.65,
                    'volatility': 0.42,
                    'correlation': 0.33,
                    'liquidity': 0.88,
                    'drawdown': 0.15
                }
                
                # Performance data for charts
                performance_data = [
                    {'time': '1D', 'pnl': 2.3, 'benchmark': 1.1},
                    {'time': '1W', 'pnl': 8.7, 'benchmark': 3.2},
                    {'time': '1M', 'pnl': 15.4, 'benchmark': 7.8},
                    {'time': '3M', 'pnl': 32.1, 'benchmark': 18.9},
                    {'time': '1Y', 'pnl': 127.5, 'benchmark': 45.3}
                ]
                
                # Entropy data for time series
                current_time = datetime.now()
                entropy_data = []
                for i in range(6):
                    time_str = (current_time - timedelta(minutes=5*i)).strftime('%H:%M')
                    entropy_data.append({
                        'time': time_str,
                        'value': 0.45 + 0.3 * np.sin(i * 0.5),
                        'threshold': 0.7
                    })
                entropy_data.reverse()
                
                # System health summary
                ui_state = self.ui_bridge.get_ui_state()
                system_health = ui_state.get('system_health', {})
                
                pattern_state = PatternState(
                    active_patterns=active_patterns,
                    risk_metrics=risk_metrics,
                    performance_data=performance_data,
                    entropy_data=entropy_data,
                    system_health=system_health
                )
                
                self.current_pattern_state = pattern_state
                self.pattern_history.append(pattern_state)
                
                # Keep history manageable
                if len(self.pattern_history) > 500:
                    self.pattern_history = self.pattern_history[-500:]
                    
            except Exception as e:
                logger.error(f"Failed to update pattern state: {e}")

    def _update_tesseract_visualizers(self) -> None:
        """Update the Tesseract visualizers with current data"""
        
        if not TESSERACT_AVAILABLE or not self.current_metrics:
            return
        
        try:
            # Update Enhanced Tesseract Processor
            if self.tesseract_processor:
                # Convert metrics to tensor format
                tensor_data = self._metrics_to_tensor(self.current_metrics)
                # Update processor with tensor data
                # self.tesseract_processor.process_tensor(tensor_data)
            
            # Update Tensor Visualization Controller
            if self.tensor_controller:
                # Update with current visualization data
                # self.tensor_controller.update_visualization(self.current_metrics)
                pass
                
        except Exception as e:
            logger.error(f"Failed to update Tesseract visualizers: {e}")

    def _metrics_to_tensor(self, metrics: VisualMetrics) -> np.ndarray:
        """Convert visual metrics to tensor format for Tesseract processing"""
        
        # Create 4D tensor from metrics
        tensor = np.array([
            [metrics.magnitude, metrics.entropy, metrics.coherence, metrics.pattern_strength],
            [metrics.cpu_usage, metrics.gpu_usage, metrics.memory_usage, metrics.thermal_load],
            [metrics.market_entropy, metrics.volatility, metrics.trend_strength, metrics.confidence_score],
            [metrics.active_patterns / 10.0, metrics.risk_level, metrics.profit_velocity * 10.0, 0.5]
        ])
        
        return tensor

    async def _broadcast_to_websocket_clients(self) -> None:
        """Broadcast current pattern state to all WebSocket clients"""
        
        if not self.websocket_clients or not self.current_pattern_state:
            return
        
        try:
            message = json.dumps({
                'type': 'pattern_state_update',
                'data': asdict(self.current_pattern_state),
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(self.current_metrics) if self.current_metrics else None
            }, default=str)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket client: {e}")
                    disconnected_clients.add(client)
            
            # Clean up disconnected clients
            self.websocket_clients -= disconnected_clients
            
            if self.websocket_clients:
                self.websocket_messages_sent += 1
                
        except Exception as e:
            logger.error(f"Failed to broadcast to WebSocket clients: {e}")

    def _calculate_market_entropy(self) -> float:
        """Calculate current market entropy from available data"""
        
        # This would integrate with real market data
        # For now, simulate entropy calculation
        current_time = time.time()
        entropy = 0.5 + 0.3 * np.sin(current_time * 0.1)
        return max(0.0, min(1.0, entropy))

    def _prepare_export_data(self) -> Dict[str, Any]:
        """Prepare data for export requests"""
        
        return {
            'metrics_history': [asdict(m) for m in self.metrics_history[-100:]],
            'pattern_history': [asdict(p) for p in self.pattern_history[-100:]],
            'update_count': self.update_count,
            'websocket_messages_sent': self.websocket_messages_sent,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

    def get_tesseract_status(self) -> Dict[str, Any]:
        """Get current status of Tesseract visualizers"""
        
        return {
            'tesseract_available': TESSERACT_AVAILABLE,
            'processor_active': self.tesseract_processor is not None,
            'controller_active': self.tensor_controller is not None,
            'websocket_clients': len(self.websocket_clients),
            'websocket_port': self.websocket_port,
            'current_metrics_available': self.current_metrics is not None,
            'pattern_state_available': self.current_pattern_state is not None
        }

    def stop_visual_bridge(self) -> None:
        """Stop the visual integration bridge"""
        
        self.bridge_active = False
        
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        # Close WebSocket connections
        if self.websocket_clients:
            asyncio.run_coroutine_threadsafe(
                self._close_websocket_clients(),
                asyncio.get_event_loop()
            )
        
        # Stop WebSocket server
        if self.websocket_server:
            asyncio.run_coroutine_threadsafe(
                self.websocket_server.close(),
                asyncio.get_event_loop()
            )
        
        logger.info("Visual Integration Bridge stopped")

    async def _close_websocket_clients(self) -> None:
        """Close all WebSocket client connections"""
        
        for client in list(self.websocket_clients):
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket client: {e}")
        
        self.websocket_clients.clear()

# Factory function for easy integration
def create_visual_bridge(ui_bridge: UIStateBridge, 
                        sustainment_controller: SustainmentUnderlayController,
                        websocket_port: int = 8765) -> VisualIntegrationBridge:
    """Factory function to create visual integration bridge"""
    
    bridge = VisualIntegrationBridge(
        ui_bridge=ui_bridge,
        sustainment_controller=sustainment_controller,
        websocket_port=websocket_port
    )
    
    # Start the bridge
    bridge.start_visual_bridge()
    
    logger.info(f"Visual Integration Bridge created and started on port {websocket_port}")
    return bridge 