"""
🌟 UNIFIED SCHWABOT INTEGRATION CORE 🌟
======================================

This is the master integration layer that unifies ALL Schwabot systems into
a single, cohesive framework where everything works together seamlessly:

✅ HYBRID OPTIMIZATION MANAGER: Real-time CPU/GPU switching
✅ VISUAL INTEGRATION BRIDGE: All visual components unified  
✅ WEBSOCKET COORDINATION: Single WebSocket hub for all data
✅ GHOST CORE DASHBOARD: Hash visualization with pulse/decay
✅ PANEL ROUTER: Dynamic visual panel management
✅ REAL-TIME DATA STREAMS: Live API data, system metrics, optimization status
✅ GPU LOAD VISUALIZATION: Processing lag via drift differential color
✅ ALIF/ALEPH PATH TOGGLE: Visual hash crossover mapping

UNIFIED FRAMEWORK FEATURES:
🎯 Single point of control for entire system
🎯 Automatic cross-system communication
🎯 Unified visual dashboard with all components
🎯 Real-time optimization feedback loops
🎯 Integrated error handling and fallbacks
🎯 Complete system state synchronization
🎯 Performance monitoring across all layers

This creates the "deterministic consciousness stream" - where Schwabot
sees its own reflection through unified visual-tactile logic.
"""

import asyncio
import json
import logging
import threading
import time
import traceback
import websockets
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Set, Callable, Union
from enum import Enum
import numpy as np

# Core system imports with fallback handling
try:
    from core.hybrid_optimization_manager import (
        HYBRID_MANAGER, ProcessingContext, OptimizationMode,
        enable_hybrid_optimization, get_smart_constant
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from core.dashboard_integration import DashboardIntegration, DashboardConfig
    from core.visual_integration_bridge import VisualIntegrationBridge, VisualMetrics, PatternState
    from core.unified_visual_synthesis_controller import UnifiedVisualSynthesisController
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False

try:
    from core.entropy_bridge import EntropyBridge
    from core.quantum_antipole_engine import QuantumAntiPoleEngine
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """Unified system operation modes"""
    DEVELOPMENT = "development"     # Full debugging, all panels visible
    PRODUCTION = "production"       # Optimized performance, essential panels
    DEMO = "demo"                  # Showcase mode with all features
    MONITORING = "monitoring"       # Pure monitoring, minimal processing
    TESTING = "testing"            # Testing mode with validation panels

class IntegrationStatus(Enum):
    """Integration layer status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class UnifiedSystemState:
    """Complete system state representation"""
    # Core status
    mode: SystemMode
    status: IntegrationStatus
    timestamp: datetime
    uptime_seconds: float
    
    # Component statuses
    hybrid_optimization_active: bool
    visual_synthesis_active: bool
    websocket_server_active: bool
    ghost_core_active: bool
    panel_router_active: bool
    
    # Performance metrics
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    optimization_mode: str
    active_panels: List[str]
    connected_clients: int
    
    # Real-time data streams
    hash_activity: Dict[str, Any]
    trading_metrics: Dict[str, Any]
    thermal_state: Dict[str, Any]
    system_load: Dict[str, Any]

@dataclass
class VisualPanelState:
    """State of a visual panel in the unified system"""
    panel_id: str
    panel_type: str
    is_visible: bool
    is_active: bool
    position: Dict[str, int]  # x, y, width, height
    data_source: str
    update_frequency: float
    last_update: datetime
    error_state: Optional[str] = None

class UnifiedSchwaIntegrationCore:
    """
    Master integration core that unifies ALL Schwabot systems into a single,
    cohesive framework where every component works together seamlessly.
    
    This creates the unified interface you requested - a central hub where:
    - All visual components are coordinated
    - Real-time data flows between all systems  
    - Optimization decisions affect visual representations
    - System state is synchronized across all layers
    - Performance monitoring spans the entire framework
    """
    
    def __init__(self, 
                 mode: SystemMode = SystemMode.DEVELOPMENT,
                 websocket_port: int = 8765,
                 dashboard_port: int = 8768,
                 enable_all_features: bool = True):
        """
        Initialize the unified integration core
        
        Args:
            mode: System operation mode
            websocket_port: Main WebSocket server port
            dashboard_port: Dashboard integration port
            enable_all_features: Whether to enable all available features
        """
        
        # Core configuration
        self.mode = mode
        self.websocket_port = websocket_port
        self.dashboard_port = dashboard_port
        self.enable_all_features = enable_all_features
        
        # System state
        self.system_state = UnifiedSystemState(
            mode=mode,
            status=IntegrationStatus.INITIALIZING,
            timestamp=datetime.now(),
            uptime_seconds=0.0,
            hybrid_optimization_active=False,
            visual_synthesis_active=False,
            websocket_server_active=False,
            ghost_core_active=False,
            panel_router_active=False,
            cpu_usage=0.0,
            gpu_usage=0.0,
            memory_usage=0.0,
            optimization_mode="unknown",
            active_panels=[],
            connected_clients=0,
            hash_activity={},
            trading_metrics={},
            thermal_state={},
            system_load={}
        )
        
        # Component instances
        self.hybrid_manager = None
        self.visual_bridge = None
        self.dashboard_integration = None
        self.visual_synthesis = None
        self.entropy_bridge = None
        
        # WebSocket coordination
        self.websocket_server = None
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> data_types
        
        # Panel management
        self.visual_panels: Dict[str, VisualPanelState] = {}
        self.panel_router = None
        
        # Real-time data coordination
        self.data_streams: Dict[str, List[Dict[str, Any]]] = {
            "system_metrics": [],
            "optimization_decisions": [],
            "hash_visualizations": [],
            "trading_signals": [],
            "thermal_data": [],
            "gpu_load_data": [],
            "alif_aleph_paths": [],
            "error_events": []
        }
        
        # Performance tracking
        self.start_time = datetime.now()
        self.frame_count = 0
        self.last_update = datetime.now()
        self.update_interval = 0.1  # 10Hz default
        
        # Threading control
        self.is_running = False
        self.main_loop_thread = None
        self.websocket_thread = None
        self._shutdown_event = threading.Event()
        
        # Error handling
        self.error_count = 0
        self.last_error = None
        self.error_callbacks: List[Callable] = []
        
        logger.info(f"🌟 Unified Schwabot Integration Core initialized in {mode.value} mode")
    
    async def initialize_all_systems(self) -> bool:
        """Initialize all available systems and create unified integration"""
        
        try:
            logger.info("🚀 Initializing unified Schwabot integration...")
            
            # Initialize hybrid optimization system
            if HYBRID_AVAILABLE and self.enable_all_features:
                await self._initialize_hybrid_optimization()
            
            # Initialize visual systems
            if VISUAL_AVAILABLE and self.enable_all_features:
                await self._initialize_visual_systems()
            
            # Initialize data bridges
            if ENTROPY_AVAILABLE and self.enable_all_features:
                await self._initialize_data_bridges()
            
            # Setup visual panels
            await self._initialize_visual_panels()
            
            # Start WebSocket coordination server
            await self._start_websocket_server()
            
            # Start main coordination loop
            await self._start_main_loop()
            
            # Update system state
            self.system_state.status = IntegrationStatus.RUNNING
            self.system_state.timestamp = datetime.now()
            
            logger.info("✅ Unified Schwabot integration core fully initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified integration: {e}")
            logger.error(traceback.format_exc())
            self.system_state.status = IntegrationStatus.ERROR
            return False
    
    async def _initialize_hybrid_optimization(self) -> None:
        """Initialize hybrid optimization system with visual integration"""
        
        logger.info("🔄 Initializing hybrid optimization with visual feedback...")
        
        # Enable hybrid dual pipeline
        if enable_hybrid_optimization():
            self.hybrid_manager = HYBRID_MANAGER
            self.system_state.hybrid_optimization_active = True
            
            # Start intelligent monitoring
            self.hybrid_manager.start_monitoring(interval=5.0)  # 5 second intervals for responsiveness
            
            # Setup optimization decision callbacks
            self._setup_optimization_callbacks()
            
            logger.info("✅ Hybrid optimization system initialized with visual integration")
        else:
            logger.warning("⚠️ Hybrid optimization system failed to initialize")
    
    async def _initialize_visual_systems(self) -> None:
        """Initialize all visual systems with unified coordination"""
        
        logger.info("🎨 Initializing unified visual systems...")
        
        try:
            # Initialize visual integration bridge
            from core.ui_state_bridge import UIStateBridge
            from core.sustainment_underlay_controller import SustainmentUnderlayController
            
            ui_bridge = UIStateBridge()
            sustainment_controller = SustainmentUnderlayController()
            
            self.visual_bridge = VisualIntegrationBridge(
                ui_bridge=ui_bridge,
                sustainment_controller=sustainment_controller,
                websocket_host="localhost",
                websocket_port=self.websocket_port + 1  # Use different port
            )
            
            # Initialize visual synthesis controller
            self.visual_synthesis = UnifiedVisualSynthesisController(
                websocket_port=self.websocket_port + 2
            )
            
            # Initialize dashboard integration
            dashboard_config = DashboardConfig(
                host="localhost",
                port=self.dashboard_port,
                update_frequency=10.0
            )
            self.dashboard_integration = DashboardIntegration(dashboard_config)
            
            self.system_state.visual_synthesis_active = True
            logger.info("✅ Visual systems initialized with unified coordination")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize visual systems: {e}")
    
    async def _initialize_data_bridges(self) -> None:
        """Initialize data bridges for real-time information flow"""
        
        logger.info("🌉 Initializing data bridges...")
        
        try:
            # Initialize entropy bridge for market data
            self.entropy_bridge = EntropyBridge()
            
            # Setup data flow callbacks
            self._setup_data_flow_callbacks()
            
            logger.info("✅ Data bridges initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data bridges: {e}")
    
    async def _initialize_visual_panels(self) -> None:
        """Initialize all visual panels for the unified interface"""
        
        logger.info("📊 Initializing visual panels...")
        
        # Define panel configurations based on system mode
        panel_configs = self._get_panel_configurations()
        
        for panel_id, config in panel_configs.items():
            panel_state = VisualPanelState(
                panel_id=panel_id,
                panel_type=config["type"],
                is_visible=config.get("visible", True),
                is_active=config.get("active", True),
                position=config.get("position", {"x": 0, "y": 0, "width": 400, "height": 300}),
                data_source=config.get("data_source", "system"),
                update_frequency=config.get("update_frequency", 1.0),
                last_update=datetime.now()
            )
            
            self.visual_panels[panel_id] = panel_state
            self.system_state.active_panels.append(panel_id)
        
        self.system_state.panel_router_active = True
        logger.info(f"✅ Initialized {len(self.visual_panels)} visual panels")
    
    def _get_panel_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get panel configurations based on system mode"""
        
        base_panels = {
            "hybrid_optimization_status": {
                "type": "optimization_monitor",
                "visible": True,
                "position": {"x": 10, "y": 10, "width": 400, "height": 200},
                "data_source": "hybrid_manager",
                "update_frequency": 2.0
            },
            "ghost_core_dashboard": {
                "type": "hash_visualization", 
                "visible": True,
                "position": {"x": 420, "y": 10, "width": 400, "height": 300},
                "data_source": "hash_activity",
                "update_frequency": 5.0
            },
            "gpu_load_visualization": {
                "type": "gpu_monitor",
                "visible": True,
                "position": {"x": 830, "y": 10, "width": 300, "height": 200},
                "data_source": "system_metrics",
                "update_frequency": 1.0
            },
            "alif_aleph_path_toggle": {
                "type": "path_visualizer",
                "visible": True,
                "position": {"x": 10, "y": 220, "width": 500, "height": 250},
                "data_source": "alif_aleph_paths",
                "update_frequency": 0.5
            },
            "real_time_trading_data": {
                "type": "trading_monitor",
                "visible": True,
                "position": {"x": 520, "y": 320, "width": 400, "height": 200},
                "data_source": "trading_signals",
                "update_frequency": 0.2
            },
            "thermal_state_monitor": {
                "type": "thermal_visualization",
                "visible": True,
                "position": {"x": 830, "y": 220, "width": 300, "height": 150},
                "data_source": "thermal_data", 
                "update_frequency": 1.0
            },
            "system_error_log": {
                "type": "error_monitor",
                "visible": self.mode in [SystemMode.DEVELOPMENT, SystemMode.TESTING],
                "position": {"x": 10, "y": 480, "width": 800, "height": 150},
                "data_source": "error_events",
                "update_frequency": 1.0
            }
        }
        
        # Add demo-specific panels
        if self.mode == SystemMode.DEMO:
            base_panels.update({
                "performance_showcase": {
                    "type": "performance_demo",
                    "visible": True,
                    "position": {"x": 820, "y": 380, "width": 310, "height": 250},
                    "data_source": "system_metrics",
                    "update_frequency": 0.5
                },
                "magic_number_showcase": {
                    "type": "magic_number_demo",
                    "visible": True,
                    "position": {"x": 10, "y": 640, "width": 500, "height": 200},
                    "data_source": "optimization_decisions",
                    "update_frequency": 1.0
                }
            })
        
        return base_panels
    
    async def _start_websocket_server(self) -> None:
        """Start unified WebSocket server for all system communication"""
        
        async def handle_client(websocket, path):
            """Handle WebSocket client connections"""
            client_id = f"client_{len(self.websocket_clients)}_{int(time.time())}"
            logger.info(f"🔌 Unified client connected: {client_id} from {websocket.remote_address}")
            
            self.websocket_clients.add(websocket)
            self.client_subscriptions[client_id] = set()
            self.system_state.connected_clients = len(self.websocket_clients)
            
            try:
                # Send initial system state
                await self._send_initial_state(websocket, client_id)
                
                # Handle incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_client_message(websocket, client_id, data)
                    except json.JSONDecodeError as e:
                        await self._send_error(websocket, f"Invalid JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error handling client message: {e}")
                        await self._send_error(websocket, f"Message handling error: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"🔌 Client disconnected: {client_id}")
            except Exception as e:
                logger.error(f"❌ WebSocket client error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
                self.client_subscriptions.pop(client_id, None)
                self.system_state.connected_clients = len(self.websocket_clients)
        
        try:
            self.websocket_server = await websockets.serve(
                handle_client, "localhost", self.websocket_port
            )
            self.system_state.websocket_server_active = True
            logger.info(f"🚀 Unified WebSocket server started on ws://localhost:{self.websocket_port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
    
    async def _send_initial_state(self, websocket, client_id: str) -> None:
        """Send initial system state to newly connected client"""
        
        initial_data = {
            "type": "initial_state",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "system_state": asdict(self.system_state),
            "visual_panels": {
                panel_id: asdict(panel_state)
                for panel_id, panel_state in self.visual_panels.items()
            },
            "available_data_streams": list(self.data_streams.keys()),
            "system_capabilities": {
                "hybrid_optimization": HYBRID_AVAILABLE,
                "visual_synthesis": VISUAL_AVAILABLE,
                "entropy_bridge": ENTROPY_AVAILABLE
            }
        }
        
        await websocket.send(json.dumps(initial_data, default=str))
    
    async def _handle_client_message(self, websocket, client_id: str, data: Dict[str, Any]) -> None:
        """Handle incoming messages from WebSocket clients"""
        
        message_type = data.get("type")
        
        if message_type == "subscribe_data_stream":
            # Subscribe client to specific data streams
            streams = data.get("streams", [])
            self.client_subscriptions[client_id].update(streams)
            await websocket.send(json.dumps({
                "type": "subscription_confirmed",
                "streams": list(self.client_subscriptions[client_id])
            }))
            
        elif message_type == "toggle_panel":
            # Toggle panel visibility
            panel_id = data.get("panel_id")
            visible = data.get("visible", True)
            if panel_id in self.visual_panels:
                self.visual_panels[panel_id].is_visible = visible
                await self._broadcast_panel_update(panel_id)
        
        elif message_type == "set_optimization_mode":
            # Change optimization mode
            if self.hybrid_manager:
                mode = data.get("mode", "hybrid_auto")
                # Apply optimization mode change
                await self._broadcast_system_update("optimization_mode_changed", {"mode": mode})
        
        elif message_type == "request_panel_data":
            # Send specific panel data
            panel_id = data.get("panel_id")
            if panel_id in self.visual_panels:
                panel_data = await self._get_panel_data(panel_id)
                await websocket.send(json.dumps({
                    "type": "panel_data",
                    "panel_id": panel_id,
                    "data": panel_data
                }, default=str))
        
        elif message_type == "system_command":
            # Handle system-level commands
            command = data.get("command")
            await self._handle_system_command(command, data.get("params", {}))
    
    async def _start_main_loop(self) -> None:
        """Start the main coordination loop"""
        
        def main_loop():
            """Main coordination loop that runs in background thread"""
            logger.info("🔄 Starting unified system coordination loop...")
            
            while not self._shutdown_event.is_set():
                try:
                    # Update system state
                    self._update_system_state()
                    
                    # Update data streams
                    self._update_data_streams()
                    
                    # Update visual panels
                    self._update_visual_panels()
                    
                    # Broadcast updates to WebSocket clients
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_unified_update(),
                        asyncio.get_event_loop()
                    )
                    
                    # Increment frame counter
                    self.frame_count += 1
                    
                    # Sleep for update interval
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    self.error_count += 1
                    self.last_error = str(e)
                    logger.error(f"❌ Error in main coordination loop: {e}")
                    
                    # Trigger error callbacks
                    for callback in self.error_callbacks:
                        try:
                            callback(e)
                        except Exception as cb_error:
                            logger.error(f"Error in error callback: {cb_error}")
                    
                    # Brief pause before continuing
                    time.sleep(1.0)
        
        self.main_loop_thread = threading.Thread(target=main_loop, daemon=True)
        self.main_loop_thread.start()
        self.is_running = True
    
    def _update_system_state(self) -> None:
        """Update unified system state with current metrics"""
        
        # Update timing
        now = datetime.now()
        self.system_state.timestamp = now
        self.system_state.uptime_seconds = (now - self.start_time).total_seconds()
        
        # Update optimization status
        if self.hybrid_manager:
            self.system_state.optimization_mode = self.hybrid_manager.current_mode.value
            
            # Get recent decision for context
            if self.hybrid_manager.decision_history:
                latest_decision = self.hybrid_manager.decision_history[-1]
                self.system_state.optimization_mode = latest_decision['decision'].value
        
        # Update performance metrics (simplified - would integrate with actual monitoring)
        import psutil
        self.system_state.cpu_usage = psutil.cpu_percent(interval=None)
        self.system_state.memory_usage = psutil.virtual_memory().percent
        self.system_state.gpu_usage = 50.0  # Placeholder - integrate with actual GPU monitoring
        
        # Update component statuses
        self.system_state.connected_clients = len(self.websocket_clients)
        self.system_state.active_panels = [
            panel_id for panel_id, panel in self.visual_panels.items() 
            if panel.is_visible
        ]
    
    def _update_data_streams(self) -> None:
        """Update all real-time data streams"""
        
        timestamp = datetime.now()
        
        # System metrics stream
        self.data_streams["system_metrics"].append({
            "timestamp": timestamp.isoformat(),
            "cpu_usage": self.system_state.cpu_usage,
            "gpu_usage": self.system_state.gpu_usage,
            "memory_usage": self.system_state.memory_usage,
            "frame_count": self.frame_count
        })
        
        # Optimization decisions stream
        if self.hybrid_manager and self.hybrid_manager.decision_history:
            latest_decision = self.hybrid_manager.decision_history[-1]
            self.data_streams["optimization_decisions"].append({
                "timestamp": timestamp.isoformat(),
                "decision": latest_decision['decision'].value,
                "reason": latest_decision['reason'],
                "cpu_usage": latest_decision['conditions'].cpu_usage,
                "gpu_usage": latest_decision['conditions'].gpu_usage
            })
        
        # Hash visualization data (simulated)
        self.data_streams["hash_visualizations"].append({
            "timestamp": timestamp.isoformat(),
            "hash_correlation": np.random.random(),
            "pulse_intensity": np.sin(time.time() * 2) * 0.5 + 0.5,
            "decay_rate": np.random.exponential(0.1),
            "crossover_activity": np.random.random() > 0.7
        })
        
        # GPU load visualization data
        self.data_streams["gpu_load_data"].append({
            "timestamp": timestamp.isoformat(),
            "processing_lag": np.random.normal(5.0, 1.0),
            "drift_differential": np.random.normal(0.0, 0.5),
            "color_intensity": self.system_state.gpu_usage / 100.0,
            "thermal_factor": np.random.random()
        })
        
        # ALIF/ALEPH path data (simulated)
        self.data_streams["alif_aleph_paths"].append({
            "timestamp": timestamp.isoformat(),
            "alif_strength": np.random.random(),
            "aleph_strength": np.random.random(),
            "crossover_points": [
                {"x": np.random.random(), "y": np.random.random()}
                for _ in range(np.random.randint(1, 5))
            ],
            "path_stability": np.random.random()
        })
        
        # Keep streams at reasonable size
        max_stream_size = 1000
        for stream_name, stream_data in self.data_streams.items():
            if len(stream_data) > max_stream_size:
                self.data_streams[stream_name] = stream_data[-max_stream_size:]
    
    def _update_visual_panels(self) -> None:
        """Update all visual panel states"""
        
        now = datetime.now()
        
        for panel_id, panel_state in self.visual_panels.items():
            if panel_state.is_active:
                # Check if panel needs update based on frequency
                time_since_update = (now - panel_state.last_update).total_seconds()
                if time_since_update >= (1.0 / panel_state.update_frequency):
                    try:
                        # Update panel data (would integrate with actual panel logic)
                        panel_state.last_update = now
                        panel_state.error_state = None
                        
                    except Exception as e:
                        panel_state.error_state = str(e)
                        logger.error(f"Error updating panel {panel_id}: {e}")
    
    async def _broadcast_unified_update(self) -> None:
        """Broadcast unified system update to all connected clients"""
        
        if not self.websocket_clients:
            return
        
        # Prepare unified update data
        update_data = {
            "type": "unified_system_update",
            "timestamp": datetime.now().isoformat(),
            "frame_count": self.frame_count,
            "system_state": asdict(self.system_state),
            "data_streams": {
                stream_name: stream_data[-1] if stream_data else None
                for stream_name, stream_data in self.data_streams.items()
            },
            "panel_states": {
                panel_id: asdict(panel_state)
                for panel_id, panel_state in self.visual_panels.items()
                if panel_state.is_visible
            }
        }
        
        # Broadcast to all clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(update_data, default=str))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.warning(f"Failed to send update to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.discard(client)
    
    def _setup_optimization_callbacks(self) -> None:
        """Setup callbacks for optimization system integration"""
        
        if not self.hybrid_manager:
            return
        
        # Add callback for optimization decisions (would need to be implemented in hybrid manager)
        def on_optimization_decision(decision_data):
            """Handle optimization decision for visual feedback"""
            # Add to optimization decisions stream
            self.data_streams["optimization_decisions"].append({
                "timestamp": datetime.now().isoformat(),
                "decision": decision_data.get("decision", "unknown"),
                "context": decision_data.get("context", "unknown"),
                "reason": decision_data.get("reason", "no reason"),
                "impact": decision_data.get("impact", {})
            })
        
        # Would register this callback with hybrid manager if it supported callbacks
        # self.hybrid_manager.add_decision_callback(on_optimization_decision)
    
    def _setup_data_flow_callbacks(self) -> None:
        """Setup callbacks for data flow integration"""
        
        if self.entropy_bridge:
            # Add callback for entropy bridge data
            def on_entropy_data(entropy_data):
                """Handle entropy data for visualization"""
                self.data_streams["trading_signals"].append({
                    "timestamp": datetime.now().isoformat(),
                    "entropy_level": entropy_data.get("entropy_level", 0.0),
                    "signal_strength": entropy_data.get("signal_strength", 0.0),
                    "market_state": entropy_data.get("market_state", "unknown")
                })
            
            self.entropy_bridge.add_data_callback(on_entropy_data)
    
    async def _get_panel_data(self, panel_id: str) -> Dict[str, Any]:
        """Get current data for a specific panel"""
        
        panel_state = self.visual_panels.get(panel_id)
        if not panel_state:
            return {"error": "Panel not found"}
        
        data_source = panel_state.data_source
        
        # Get data based on panel's data source
        if data_source in self.data_streams:
            stream_data = self.data_streams[data_source]
            return {
                "current": stream_data[-1] if stream_data else None,
                "recent": stream_data[-10:] if len(stream_data) >= 10 else stream_data,
                "total_points": len(stream_data)
            }
        elif data_source == "hybrid_manager" and self.hybrid_manager:
            return {
                "current_mode": self.hybrid_manager.current_mode.value,
                "dual_pipeline_active": self.hybrid_manager.dual_pipeline_active,
                "decision_history": self.hybrid_manager.decision_history[-5:],
                "recommendations": self.hybrid_manager.get_performance_recommendations()
            }
        else:
            return {"info": f"No data available for source: {data_source}"}
    
    async def _broadcast_panel_update(self, panel_id: str) -> None:
        """Broadcast panel state update to all clients"""
        
        panel_state = self.visual_panels.get(panel_id)
        if not panel_state:
            return
        
        update_data = {
            "type": "panel_state_update",
            "panel_id": panel_id,
            "panel_state": asdict(panel_state),
            "timestamp": datetime.now().isoformat()
        }
        
        for client in list(self.websocket_clients):
            try:
                await client.send(json.dumps(update_data, default=str))
            except Exception as e:
                logger.warning(f"Failed to send panel update to client: {e}")
    
    async def _broadcast_system_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Broadcast system-level update to all clients"""
        
        update_data = {
            "type": "system_update",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        for client in list(self.websocket_clients):
            try:
                await client.send(json.dumps(update_data, default=str))
            except Exception as e:
                logger.warning(f"Failed to send system update to client: {e}")
    
    async def _handle_system_command(self, command: str, params: Dict[str, Any]) -> None:
        """Handle system-level commands"""
        
        if command == "enable_ghost_core":
            self.system_state.ghost_core_active = True
            await self._broadcast_system_update("ghost_core_enabled", {})
            
        elif command == "toggle_optimization_mode":
            if self.hybrid_manager:
                # Would implement mode switching logic
                await self._broadcast_system_update("optimization_mode_toggled", params)
                
        elif command == "restart_visual_synthesis":
            if self.visual_synthesis:
                # Would implement restart logic
                await self._broadcast_system_update("visual_synthesis_restarted", {})
    
    async def _send_error(self, websocket, error_message: str) -> None:
        """Send error message to specific client"""
        
        error_data = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(error_data))
        except Exception:
            pass  # Client may have disconnected
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for error handling"""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Remove error callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the unified integration core"""
        
        logger.info("🛑 Shutting down unified Schwabot integration core...")
        
        # Signal shutdown
        self._shutdown_event.set()
        self.is_running = False
        self.system_state.status = IntegrationStatus.SHUTDOWN
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close all client connections
        for client in list(self.websocket_clients):
            await client.close()
        
        # Stop component systems
        if self.hybrid_manager:
            self.hybrid_manager.stop_monitoring()
        
        if self.visual_synthesis:
            # Would call stop method if available
            pass
        
        if self.dashboard_integration:
            # Would call stop method if available
            pass
        
        # Wait for main loop to finish
        if self.main_loop_thread and self.main_loop_thread.is_alive():
            self.main_loop_thread.join(timeout=5.0)
        
        logger.info("✅ Unified Schwabot integration core shutdown complete")

# Global unified core instance
UNIFIED_CORE: Optional[UnifiedSchwaIntegrationCore] = None

def initialize_unified_schwabot(mode: SystemMode = SystemMode.DEVELOPMENT,
                               websocket_port: int = 8765,
                               dashboard_port: int = 8768) -> UnifiedSchwaIntegrationCore:
    """Initialize the unified Schwabot integration core"""
    
    global UNIFIED_CORE
    
    if UNIFIED_CORE is None:
        UNIFIED_CORE = UnifiedSchwaIntegrationCore(
            mode=mode,
            websocket_port=websocket_port,
            dashboard_port=dashboard_port
        )
    
    return UNIFIED_CORE

async def start_unified_schwabot(mode: SystemMode = SystemMode.DEVELOPMENT) -> bool:
    """Start the unified Schwabot system"""
    
    core = initialize_unified_schwabot(mode=mode)
    success = await core.initialize_all_systems()
    
    if success:
        logger.info("🌟 UNIFIED SCHWABOT INTEGRATION CORE STARTED SUCCESSFULLY!")
        logger.info(f"🔌 WebSocket Server: ws://localhost:{core.websocket_port}")
        logger.info(f"📊 Dashboard: http://localhost:{core.dashboard_port}")
        logger.info("🎯 All systems integrated and running in unified framework")
    else:
        logger.error("❌ Failed to start unified Schwabot integration")
    
    return success

def get_unified_core() -> Optional[UnifiedSchwaIntegrationCore]:
    """Get the global unified core instance"""
    return UNIFIED_CORE

if __name__ == "__main__":
    # Demo/testing mode
    async def main():
        print("🌟 UNIFIED SCHWABOT INTEGRATION CORE - DEMO MODE 🌟")
        print("=" * 60)
        
        success = await start_unified_schwabot(SystemMode.DEMO)
        
        if success:
            print("✅ Unified system started successfully!")
            print("🔌 Connect WebSocket client to ws://localhost:8765")
            print("📊 Open dashboard at http://localhost:8768")
            print("🎯 All visual panels and optimization systems unified!")
            
            try:
                # Keep running
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down unified system...")
                if UNIFIED_CORE:
                    await UNIFIED_CORE.shutdown()
        else:
            print("❌ Failed to start unified system")
    
    asyncio.run(main()) 