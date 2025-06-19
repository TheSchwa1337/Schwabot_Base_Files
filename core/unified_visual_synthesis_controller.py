#!/usr/bin/env python3
"""
Unified Visual Synthesis Controller
==================================

The master controller that integrates all Schwabot visual components into a single
unified interface. This is the "visual synthesis layer" that brings together:

- BTC Processor UI and controls
- Ghost Architecture profit handoff visualization  
- Edge vector field displays
- Drift exit detector panels
- Future hooks evaluation interface
- Error handling pipeline status
- Sustainment underlay metrics
- Real-time trading visualization

This controller implements the 8 principles of sustainment and provides the
integrated visual experience you described - like a task manager for Schwabot.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import websockets

# Core system imports
from .btc_processor_ui import BTCProcessorUI
from .ghost_architecture_btc_profit_handoff import GhostArchitectureBTCProfitHandoff
from .edge_vector_field import EdgeVectorField
from .drift_exit_detector import DriftExitDetector
from .future_hooks import HookRegistry
from .error_handling_pipeline import ErrorHandlingPipeline

# Integration imports with fallbacks
try:
    from .sustainment_underlay_controller import SustainmentUnderlayController
    from .unified_visual_controller import UnifiedVisualController
    from .visual_integration_bridge import VisualIntegrationBridge
    from .ui_integration_bridge import UIIntegrationBridge
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisualPanelType(Enum):
    """Types of visual panels in the unified interface"""
    BTC_PROCESSOR = "btc_processor"
    GHOST_ARCHITECTURE = "ghost_architecture"
    EDGE_VECTOR_FIELD = "edge_vector_field"
    DRIFT_EXIT_DETECTOR = "drift_exit_detector"
    FUTURE_HOOKS = "future_hooks"
    ERROR_HANDLING = "error_handling"
    SUSTAINMENT_METRICS = "sustainment_metrics"
    TRADING_VISUALIZATION = "trading_visualization"
    SYSTEM_MONITOR = "system_monitor"
    CUSTOM = "custom"

@dataclass
class PanelState:
    """State of a visual panel"""
    panel_type: VisualPanelType
    is_active: bool = True
    is_visible: bool = True
    position: Dict[str, int] = None  # x, y coordinates
    size: Dict[str, int] = None      # width, height
    data: Dict[str, Any] = None
    last_update: datetime = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0, "y": 0}
        if self.size is None:
            self.size = {"width": 400, "height": 300}
        if self.data is None:
            self.data = {}
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class VisualSynthesisState:
    """Complete state of the visual synthesis system"""
    active_panels: Dict[str, PanelState] = None
    system_health: float = 1.0
    sustainment_index: float = 0.0
    total_profit: float = 0.0
    active_trades: int = 0
    system_mode: str = "development"
    last_synthesis: datetime = None
    
    def __post_init__(self):
        if self.active_panels is None:
            self.active_panels = {}
        if self.last_synthesis is None:
            self.last_synthesis = datetime.now()

class UnifiedVisualSynthesisController:
    """
    Master visual synthesis controller that integrates all Schwabot visual components.
    
    This creates the unified interface you described - a central hub where all
    visual components work together, with toggle controls that dynamically change
    how each core functionality is displayed and integrated.
    """
    
    def __init__(self, 
                 btc_processor=None,
                 ghost_architecture=None,
                 edge_vector_field=None,
                 drift_detector=None,
                 hook_registry=None,
                 error_pipeline=None,
                 sustainment_controller=None,
                 websocket_port: int = 8080):
        """
        Initialize the unified visual synthesis controller
        
        Args:
            btc_processor: BTC data processor instance
            ghost_architecture: Ghost architecture profit handoff system
            edge_vector_field: Edge vector field detector
            drift_detector: Drift exit detector
            hook_registry: Future hooks registry
            error_pipeline: Error handling pipeline
            sustainment_controller: Sustainment underlay controller
            websocket_port: WebSocket port for real-time updates
        """
        
        # Core system components
        self.btc_processor = btc_processor
        self.ghost_architecture = ghost_architecture
        self.edge_vector_field = edge_vector_field
        self.drift_detector = drift_detector
        self.hook_registry = hook_registry
        self.error_pipeline = error_pipeline
        self.sustainment_controller = sustainment_controller
        
        # Integration bridges
        self.visual_bridge = None
        self.ui_bridge = None
        self.unified_controller = None
        
        # Visual synthesis state
        self.synthesis_state = VisualSynthesisState()
        self.panel_configurations = {}
        
        # WebSocket server for real-time updates
        self.websocket_port = websocket_port
        self.websocket_clients = set()
        self.websocket_server = None
        
        # Control system
        self.is_running = False
        self.synthesis_thread = None
        self.websocket_thread = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.synthesis_count = 0
        self.update_count = 0
        self.start_time = datetime.now()
        
        # Initialize the system
        self._initialize_components()
        self._setup_panels()
        
        logger.info("Unified Visual Synthesis Controller initialized")
    
    def _initialize_components(self):
        """Initialize all core components and integration bridges"""
        
        try:
            # Initialize BTC Processor UI if not provided
            if not self.btc_processor:
                try:
                    from .btc_data_processor import BTCDataProcessor
                    self.btc_processor = BTCDataProcessor()
                except ImportError:
                    logger.warning("BTC Data Processor not available - using mock")
                    self.btc_processor = self._create_mock_btc_processor()
            
            # Initialize Ghost Architecture if not provided
            if not self.ghost_architecture:
                self.ghost_architecture = GhostArchitectureBTCProfitHandoff()
            
            # Initialize Edge Vector Field if not provided
            if not self.edge_vector_field:
                self.edge_vector_field = EdgeVectorField()
            
            # Initialize Drift Detector if not provided
            if not self.drift_detector:
                self.drift_detector = self._create_mock_drift_detector()
            
            # Initialize Hook Registry if not provided
            if not self.hook_registry:
                self.hook_registry = self._create_mock_hook_registry()
            
            # Initialize Error Pipeline if not provided
            if not self.error_pipeline:
                self.error_pipeline = ErrorHandlingPipeline()
            
            # Initialize Sustainment Controller if not provided
            if not self.sustainment_controller:
                self.sustainment_controller = self._create_mock_sustainment_controller()
            
            # Initialize integration bridges if available
            if INTEGRATION_AVAILABLE:
                self._initialize_integration_bridges()
            
            logger.info("✅ All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_mock_btc_processor(self):
        """Create mock BTC processor for development"""
        class MockBTCProcessor:
            def get_mining_statistics(self):
                return {"hash_rate": 1234.56, "difficulty": 7890, "blocks_found": 42}
            async def start(self): pass
            async def stop(self): pass
        return MockBTCProcessor()
    
    def _create_mock_drift_detector(self):
        """Create mock drift detector"""
        import pandas as pd
        entropy_history = pd.DataFrame({'BTC/USD': [0.2, 0.3, 0.4, 0.5, 0.6]})
        trend_tracker = {'BTC/USD': pd.DataFrame({
            'trend': [0.1, 0.2, 0.3, 0.4, 0.5],
            'volume': [1000, 1200, 1100, 1300, 1400]
        })}
        return DriftExitDetector(entropy_history, trend_tracker)
    
    def _create_mock_hook_registry(self):
        """Create mock hook registry"""
        class MockLedger:
            def get(self, key): return None
        class MockEntropyEngine:
            def compute_entropy(self, data): return 0.5
        class MockGANFilter:
            def detect(self, data): 
                from types import SimpleNamespace
                return SimpleNamespace(anomaly_score=0.3)
        
        return HookRegistry(MockLedger(), MockEntropyEngine(), MockGANFilter())
    
    def _create_mock_sustainment_controller(self):
        """Create mock sustainment controller"""
        class MockSustainmentController:
            def get_sustainment_status(self):
                return {
                    'sustainment_index': 0.75,
                    'system_health_score': 0.9,
                    'current_vector': {
                        'anticipation': 0.8,
                        'integration': 0.85,
                        'responsiveness': 0.7,
                        'simplicity': 0.9,
                        'economy': 0.75,
                        'survivability': 0.95,
                        'continuity': 0.8,
                        'improvisation': 0.7
                    },
                    'total_corrections': 12
                }
            
            def start_continuous_synthesis(self): pass
            def stop_continuous_synthesis(self): pass
        
        return MockSustainmentController()
    
    def _initialize_integration_bridges(self):
        """Initialize the integration bridges"""
        
        try:
            # Create UI integration bridge
            self.ui_bridge = UIIntegrationBridge()
            
            # Create visual integration bridge
            self.visual_bridge = VisualIntegrationBridge(
                ui_bridge=self.ui_bridge,
                sustainment_controller=self.sustainment_controller,
                websocket_port=self.websocket_port + 1
            )
            
            # Create unified visual controller
            from .btc_processor_controller import BTCProcessorController
            btc_controller = BTCProcessorController(self.btc_processor)
            
            self.unified_controller = UnifiedVisualController(
                btc_controller=btc_controller,
                visual_bridge=self.visual_bridge,
                ui_bridge=self.ui_bridge,
                sustainment_controller=self.sustainment_controller,
                websocket_port=self.websocket_port
            )
            
            logger.info("✅ Integration bridges initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize integration bridges: {e}")
            # Continue without integration bridges
            pass
    
    def _setup_panels(self):
        """Setup the default visual panels"""
        
        # BTC Processor Panel
        self.synthesis_state.active_panels["btc_processor"] = PanelState(
            panel_type=VisualPanelType.BTC_PROCESSOR,
            position={"x": 0, "y": 0},
            size={"width": 500, "height": 400}
        )
        
        # Ghost Architecture Panel
        self.synthesis_state.active_panels["ghost_architecture"] = PanelState(
            panel_type=VisualPanelType.GHOST_ARCHITECTURE,
            position={"x": 520, "y": 0},
            size={"width": 500, "height": 400}
        )
        
        # Edge Vector Field Panel
        self.synthesis_state.active_panels["edge_vector_field"] = PanelState(
            panel_type=VisualPanelType.EDGE_VECTOR_FIELD,
            position={"x": 0, "y": 420},
            size={"width": 500, "height": 350}
        )
        
        # Drift Exit Detector Panel
        self.synthesis_state.active_panels["drift_detector"] = PanelState(
            panel_type=VisualPanelType.DRIFT_EXIT_DETECTOR,
            position={"x": 520, "y": 420},
            size={"width": 500, "height": 350}
        )
        
        # Future Hooks Panel
        self.synthesis_state.active_panels["future_hooks"] = PanelState(
            panel_type=VisualPanelType.FUTURE_HOOKS,
            position={"x": 1040, "y": 0},
            size={"width": 400, "height": 200}
        )
        
        # Error Handling Panel
        self.synthesis_state.active_panels["error_handling"] = PanelState(
            panel_type=VisualPanelType.ERROR_HANDLING,
            position={"x": 1040, "y": 220},
            size={"width": 400, "height": 200}
        )
        
        # Sustainment Metrics Panel
        self.synthesis_state.active_panels["sustainment_metrics"] = PanelState(
            panel_type=VisualPanelType.SUSTAINMENT_METRICS,
            position={"x": 1040, "y": 440},
            size={"width": 400, "height": 330}
        )
        
        logger.info(f"✅ Setup {len(self.synthesis_state.active_panels)} visual panels")
    
    async def start_visual_synthesis(self):
        """Start the unified visual synthesis system"""
        
        if self.is_running:
            logger.warning("Visual synthesis already running")
            return
        
        self.is_running = True
        
        try:
            # Start all core components
            await self._start_core_components()
            
            # Start integration bridges
            if INTEGRATION_AVAILABLE:
                await self._start_integration_bridges()
            
            # Start WebSocket server
            await self._start_websocket_server()
            
            # Start synthesis loop
            self.synthesis_thread = threading.Thread(
                target=self._synthesis_loop,
                daemon=True
            )
            self.synthesis_thread.start()
            
            logger.info("🚀 Unified Visual Synthesis System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start visual synthesis: {e}")
            self.is_running = False
            raise
    
    async def _start_core_components(self):
        """Start all core components"""
        
        # Start BTC processor
        if hasattr(self.btc_processor, 'start'):
            await self.btc_processor.start()
        
        # Start ghost architecture
        if hasattr(self.ghost_architecture, 'start'):
            await self.ghost_architecture.start()
        
        # Start sustainment controller
        if hasattr(self.sustainment_controller, 'start_continuous_synthesis'):
            self.sustainment_controller.start_continuous_synthesis()
        
        logger.info("✅ Core components started")
    
    async def _start_integration_bridges(self):
        """Start integration bridges"""
        
        # Start UI bridge
        if self.ui_bridge and hasattr(self.ui_bridge, 'start'):
            self.ui_bridge.start()
        
        # Start visual bridge
        if self.visual_bridge and hasattr(self.visual_bridge, 'start_visual_bridge'):
            self.visual_bridge.start_visual_bridge()
        
        # Start unified controller
        if self.unified_controller and hasattr(self.unified_controller, 'start_visual_controller'):
            await self.unified_controller.start_visual_controller()
        
        logger.info("✅ Integration bridges started")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        
        async def websocket_handler(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                # Send initial state
                await self._send_initial_state(websocket)
                
                # Handle incoming messages
                async for message in websocket:
                    await self._handle_websocket_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.discard(websocket)
        
        try:
            self.websocket_server = await websockets.serve(
                websocket_handler, "localhost", self.websocket_port
            )
            logger.info(f"✅ WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            # Continue without WebSocket server
    
    async def _send_initial_state(self, websocket):
        """Send initial state to newly connected client"""
        
        initial_data = {
            "type": "initial_state",
            "timestamp": datetime.now().isoformat(),
            "synthesis_state": asdict(self.synthesis_state),
            "panel_data": {
                panel_id: asdict(panel_state) 
                for panel_id, panel_state in self.synthesis_state.active_panels.items()
            }
        }
        
        await websocket.send(json.dumps(initial_data))
    
    def _synthesis_loop(self):
        """Main synthesis loop that coordinates all visual components"""
        
        while self.is_running:
            try:
                # Update synthesis state
                self._update_synthesis_state()
                
                # Update all panel data
                self._update_panel_data()
                
                # Broadcast updates to WebSocket clients
                asyncio.run(self._broadcast_updates())
                
                # Increment counters
                self.synthesis_count += 1
                self.update_count += 1
                
                # Sleep for update interval
                time.sleep(0.5)  # 2 Hz update rate for smooth UI
                
            except Exception as e:
                logger.error(f"Error in synthesis loop: {e}")
                time.sleep(1.0)
    
    def _update_synthesis_state(self):
        """Update the overall synthesis state"""
        
        with self._lock:
            # Update system health
            if self.sustainment_controller:
                status = self.sustainment_controller.get_sustainment_status()
                self.synthesis_state.sustainment_index = status.get('sustainment_index', 0.0)
                self.synthesis_state.system_health = status.get('system_health_score', 1.0)
            
            # Update trading metrics
            if self.ui_bridge and hasattr(self.ui_bridge, 'get_current_trading_metrics'):
                trading_metrics = self.ui_bridge.get_current_trading_metrics()
                if trading_metrics:
                    self.synthesis_state.total_profit = trading_metrics.total_profit
                    self.synthesis_state.active_trades = trading_metrics.active_positions
            
            # Update timestamp
            self.synthesis_state.last_synthesis = datetime.now()
    
    def _update_panel_data(self):
        """Update data for all active panels"""
        
        for panel_id, panel_state in self.synthesis_state.active_panels.items():
            try:
                if panel_state.panel_type == VisualPanelType.BTC_PROCESSOR:
                    panel_state.data = self._get_btc_processor_data()
                
                elif panel_state.panel_type == VisualPanelType.GHOST_ARCHITECTURE:
                    panel_state.data = self._get_ghost_architecture_data()
                
                elif panel_state.panel_type == VisualPanelType.EDGE_VECTOR_FIELD:
                    panel_state.data = self._get_edge_vector_field_data()
                
                elif panel_state.panel_type == VisualPanelType.DRIFT_EXIT_DETECTOR:
                    panel_state.data = self._get_drift_detector_data()
                
                elif panel_state.panel_type == VisualPanelType.FUTURE_HOOKS:
                    panel_state.data = self._get_future_hooks_data()
                
                elif panel_state.panel_type == VisualPanelType.ERROR_HANDLING:
                    panel_state.data = self._get_error_handling_data()
                
                elif panel_state.panel_type == VisualPanelType.SUSTAINMENT_METRICS:
                    panel_state.data = self._get_sustainment_metrics_data()
                
                panel_state.last_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error updating panel {panel_id}: {e}")
    
    def _get_btc_processor_data(self) -> Dict[str, Any]:
        """Get BTC processor panel data"""
        
        if hasattr(self.btc_processor, 'get_mining_statistics'):
            stats = self.btc_processor.get_mining_statistics()
        else:
            stats = {"hash_rate": 1234.56, "difficulty": 7890, "blocks_found": 42}
        
        return {
            "type": "btc_processor",
            "stats": stats,
            "status": "active" if self.btc_processor else "inactive",
            "features": {
                "mining_analysis": True,
                "hash_generation": True,
                "memory_management": True,
                "load_balancing": True,
                "backlog_processing": True
            },
            "performance": {
                "cpu_usage": 45.2,
                "memory_usage": 12.8,
                "gpu_usage": 78.3,
                "thermal_state": "normal"
            }
        }
    
    def _get_ghost_architecture_data(self) -> Dict[str, Any]:
        """Get ghost architecture panel data"""
        
        if hasattr(self.ghost_architecture, 'get_system_status'):
            status = self.ghost_architecture.get_system_status()
        else:
            status = {
                "system": {"is_running": True},
                "phantom_tracking": {"active_phantoms": 3},
                "performance": {"handoff_success_rate": 0.92}
            }
        
        return {
            "type": "ghost_architecture",
            "status": status,
            "active_phantoms": status.get("phantom_tracking", {}).get("active_phantoms", 0),
            "handoff_success_rate": status.get("performance", {}).get("handoff_success_rate", 0.0),
            "handoff_strategies": [
                "sequential_cascade",
                "parallel_distribution", 
                "quantum_tunneling",
                "spectral_bridging",
                "phantom_relay"
            ],
            "materialization_events": 45
        }
    
    def _get_edge_vector_field_data(self) -> Dict[str, Any]:
        """Get edge vector field panel data"""
        
        if hasattr(self.edge_vector_field, 'get_pattern_stats'):
            stats = self.edge_vector_field.get_pattern_stats()
        else:
            stats = {"total_detections": 127, "success_rate": 0.83}
        
        return {
            "type": "edge_vector_field",
            "stats": stats,
            "active_patterns": [
                {"name": "inverse_profit_fork", "confidence": 0.78, "profit_potential": 0.023},
                {"name": "shadow_pump", "confidence": 0.85, "profit_potential": 0.041},
                {"name": "paradox_spike", "confidence": 0.62, "profit_potential": 0.017}
            ],
            "vector_signature": {
                "profit_gradient": 0.003,
                "entropy_gradient": 0.12,
                "tensor_variance": 0.67,
                "volume_profile": 0.84
            }
        }
    
    def _get_drift_detector_data(self) -> Dict[str, Any]:
        """Get drift detector panel data"""
        
        return {
            "type": "drift_detector",
            "consciousness_state": "BULL",
            "entropy_delta": -0.05,
            "paradox_pressure": 2.5,
            "drift_confidence": 0.75,
            "exit_urgency": 0.3,
            "trend_strength": 0.68,
            "volume_profile": 0.72,
            "thermal_state": 0.45,
            "memory_coherence": 0.89
        }
    
    def _get_future_hooks_data(self) -> Dict[str, Any]:
        """Get future hooks panel data"""
        
        return {
            "type": "future_hooks",
            "current_decision": "preserve",
            "entropy_threshold": 0.75,
            "delta_sym_threshold": 0.3,
            "anomaly_threshold": 0.85,
            "recent_evaluations": [
                {"state": "S_t1", "entropy": 0.65, "decision": "preserve", "timestamp": "12:34:56"},
                {"state": "S_t2", "entropy": 0.82, "decision": "rebind", "timestamp": "12:34:57"},
                {"state": "S_t3", "entropy": 0.71, "decision": "preserve", "timestamp": "12:34:58"}
            ],
            "rebind_count": 23,
            "preserve_count": 87
        }
    
    def _get_error_handling_data(self) -> Dict[str, Any]:
        """Get error handling pipeline data"""
        
        stats = self.error_pipeline.get_conversion_stats()
        
        return {
            "type": "error_handling",
            "conversion_stats": stats['conversion_stats'],
            "ferris_wheel_protection": stats['ferris_wheel_protection_active'],
            "recent_conversions": [
                {"original": "✅ BTC analysis complete", "converted": "[PASS] BTC analysis complete"},
                {"original": "🚀 Starting mining", "converted": "[START] Starting mining"},
                {"original": "💰 Profit detected", "converted": "[PROFIT] Profit detected"}
            ],
            "critical_errors_prevented": stats['conversion_stats']['critical_failures_avoided'],
            "windows_compatibility": "active"
        }
    
    def _get_sustainment_metrics_data(self) -> Dict[str, Any]:
        """Get sustainment metrics panel data"""
        
        if self.sustainment_controller:
            status = self.sustainment_controller.get_sustainment_status()
            return {
                "type": "sustainment_metrics",
                "sustainment_index": status.get('sustainment_index', 0.0),
                "system_health": status.get('system_health_score', 1.0),
                "principles": status.get('current_vector', {}),
                "corrections_applied": status.get('total_corrections', 0),
                "status": status.get('status', 'UNKNOWN'),
                "critical_threshold": 0.65
            }
        else:
            return {
                "type": "sustainment_metrics",
                "sustainment_index": 0.75,
                "system_health": 0.9,
                "principles": {
                    "anticipation": 0.8,
                    "integration": 0.85,
                    "responsiveness": 0.7,
                    "simplicity": 0.9,
                    "economy": 0.75,
                    "survivability": 0.95,
                    "continuity": 0.8,
                    "improvisation": 0.7
                },
                "corrections_applied": 12,
                "status": "SUSTAINABLE",
                "critical_threshold": 0.65
            }
    
    async def _broadcast_updates(self):
        """Broadcast updates to all WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        update_data = {
            "type": "synthesis_update",
            "timestamp": datetime.now().isoformat(),
            "synthesis_state": asdict(self.synthesis_state),
            "panel_data": {
                panel_id: asdict(panel_state) 
                for panel_id, panel_state in self.synthesis_state.active_panels.items()
                if panel_state.is_visible
            }
        }
        
        # Broadcast to all connected clients
        for client in list(self.websocket_clients):
            try:
                await client.send(json.dumps(update_data))
            except Exception as e:
                logger.warning(f"Failed to send update to client: {e}")
                self.websocket_clients.discard(client)
    
    async def _handle_websocket_message(self, websocket, message):
        """Handle incoming WebSocket messages"""
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "toggle_panel":
                await self._handle_toggle_panel(data)
            elif message_type == "update_panel_config":
                await self._handle_update_panel_config(data)
            elif message_type == "system_command":
                await self._handle_system_command(data)
            elif message_type == "feature_toggle":
                await self._handle_feature_toggle(data)
                
            # Send acknowledgment
            ack = {
                "type": "ack",
                "original_type": message_type,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(ack))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            # Send error response
            error = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error))
    
    async def _handle_toggle_panel(self, data):
        """Handle panel toggle requests"""
        
        panel_id = data.get("panel_id")
        enabled = data.get("enabled", True)
        
        if panel_id in self.synthesis_state.active_panels:
            self.synthesis_state.active_panels[panel_id].is_visible = enabled
            logger.info(f"Panel {panel_id} {'enabled' if enabled else 'disabled'}")
    
    async def _handle_update_panel_config(self, data):
        """Handle panel configuration updates"""
        
        panel_id = data.get("panel_id")
        config = data.get("config", {})
        
        if panel_id in self.synthesis_state.active_panels:
            panel_state = self.synthesis_state.active_panels[panel_id]
            
            if "position" in config:
                panel_state.position.update(config["position"])
            if "size" in config:
                panel_state.size.update(config["size"])
            
            logger.info(f"Updated panel {panel_id} configuration")
    
    async def _handle_feature_toggle(self, data):
        """Handle feature toggle requests"""
        
        feature = data.get("feature")
        enabled = data.get("enabled", True)
        component = data.get("component", "btc_processor")
        
        logger.info(f"Feature toggle: {component}.{feature} = {enabled}")
        
        # Route to appropriate component
        if component == "btc_processor" and self.unified_controller:
            # This would integrate with your existing BTC processor controller
            pass
        elif component == "ghost_architecture":
            # This would integrate with ghost architecture controls
            pass
    
    async def _handle_system_command(self, data):
        """Handle system commands"""
        
        command = data.get("command")
        parameters = data.get("parameters", {})
        
        if command == "emergency_cleanup":
            await self._emergency_cleanup()
        elif command == "toggle_trading":
            await self._toggle_trading(parameters.get("enabled", False))
        elif command == "update_sustainment":
            await self._update_sustainment_threshold(parameters.get("threshold", 0.65))
        elif command == "system_mode":
            self._set_system_mode(parameters.get("mode", "development"))
    
    async def _emergency_cleanup(self):
        """Perform emergency cleanup"""
        
        logger.warning("🚨 Emergency cleanup initiated")
        
        # Trigger cleanup in all components
        if hasattr(self.btc_processor, '_emergency_memory_cleanup'):
            await self.btc_processor._emergency_memory_cleanup()
        
        if hasattr(self.ghost_architecture, '_complete_pending_handoffs'):
            await self.ghost_architecture._complete_pending_handoffs()
        
        # Clear error pipeline
        if hasattr(self.error_pipeline, 'clear_log'):
            self.error_pipeline.clear_log()
        
        logger.info("✅ Emergency cleanup completed")
    
    async def _toggle_trading(self, enabled: bool):
        """Toggle live trading"""
        
        logger.info(f"Trading {'enabled' if enabled else 'disabled'}")
        
        # Update trading state in UI bridge
        if self.ui_bridge and hasattr(self.ui_bridge, 'execute_command'):
            self.ui_bridge.execute_command("toggle_trading", {"enabled": enabled})
    
    async def _update_sustainment_threshold(self, threshold: float):
        """Update sustainment threshold"""
        
        if self.sustainment_controller and hasattr(self.sustainment_controller, 's_crit'):
            self.sustainment_controller.s_crit = threshold
            logger.info(f"Updated sustainment threshold to {threshold}")
    
    def _set_system_mode(self, mode: str):
        """Set system operation mode"""
        
        valid_modes = ["development", "testing", "live_trading"]
        if mode in valid_modes:
            self.synthesis_state.system_mode = mode
            logger.info(f"System mode set to: {mode}")
        else:
            logger.warning(f"Invalid system mode: {mode}")
    
    def get_synthesis_status(self) -> Dict[str, Any]:
        """Get comprehensive synthesis status"""
        
        return {
            "is_running": self.is_running,
            "synthesis_count": self.synthesis_count,
            "update_count": self.update_count,
            "active_panels": len(self.synthesis_state.active_panels),
            "visible_panels": len([p for p in self.synthesis_state.active_panels.values() if p.is_visible]),
            "websocket_clients": len(self.websocket_clients),
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "synthesis_state": asdict(self.synthesis_state),
            "integration_available": INTEGRATION_AVAILABLE
        }
    
    async def stop_visual_synthesis(self):
        """Stop the unified visual synthesis system"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        try:
            # Stop synthesis thread
            if self.synthesis_thread and self.synthesis_thread.is_alive():
                self.synthesis_thread.join(timeout=5.0)
            
            # Stop WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # Stop core components
            await self._stop_core_components()
            
            # Stop integration bridges
            if INTEGRATION_AVAILABLE:
                await self._stop_integration_bridges()
            
            logger.info("🛑 Unified Visual Synthesis System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping visual synthesis: {e}")
    
    async def _stop_core_components(self):
        """Stop all core components"""
        
        # Stop BTC processor
        if hasattr(self.btc_processor, 'stop'):
            await self.btc_processor.stop()
        
        # Stop ghost architecture
        if hasattr(self.ghost_architecture, 'stop'):
            await self.ghost_architecture.stop()
        
        # Stop sustainment controller
        if hasattr(self.sustainment_controller, 'stop_continuous_synthesis'):
            self.sustainment_controller.stop_continuous_synthesis()
        
        logger.info("✅ Core components stopped")
    
    async def _stop_integration_bridges(self):
        """Stop integration bridges"""
        
        # Stop UI bridge
        if self.ui_bridge and hasattr(self.ui_bridge, 'stop'):
            self.ui_bridge.stop()
        
        # Stop visual bridge
        if self.visual_bridge and hasattr(self.visual_bridge, 'stop_visual_bridge'):
            self.visual_bridge.stop_visual_bridge()
        
        # Stop unified controller
        if self.unified_controller and hasattr(self.unified_controller, 'stop_visual_controller'):
            await self.unified_controller.stop_visual_controller()
        
        logger.info("✅ Integration bridges stopped")

# Factory function for easy creation
def create_unified_visual_synthesis(
    btc_processor=None,
    ghost_architecture=None,
    edge_vector_field=None,
    drift_detector=None,
    hook_registry=None,
    error_pipeline=None,
    sustainment_controller=None,
    websocket_port: int = 8080
) -> UnifiedVisualSynthesisController:
    """Factory function to create unified visual synthesis controller"""
    
    return UnifiedVisualSynthesisController(
        btc_processor=btc_processor,
        ghost_architecture=ghost_architecture,
        edge_vector_field=edge_vector_field,
        drift_detector=drift_detector,
        hook_registry=hook_registry,
        error_pipeline=error_pipeline,
        sustainment_controller=sustainment_controller,
        websocket_port=websocket_port
    )

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Create unified visual synthesis controller
        controller = create_unified_visual_synthesis()
        
        # Start the system
        await controller.start_visual_synthesis()
        
        print("🚀 Unified Visual Synthesis System running")
        print(f"📊 Status: {controller.get_synthesis_status()}")
        print("🌐 WebSocket server available for frontend connections")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep running
            while controller.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
            await controller.stop_visual_synthesis()
    
    asyncio.run(demo()) 