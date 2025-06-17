#!/usr/bin/env python3
"""
Unified Visual Controller for Schwabot
=====================================

A comprehensive visual interface controller that embraces the 8 principles of sustainment:
1. Anticipation - Predictive UI states and proactive controls
2. Continuity - Seamless operation without interface disruption  
3. Responsiveness - Real-time updates and immediate feedback
4. Integration - Unified control of all system components
5. Simplicity - Clean, intuitive interface design
6. Improvisation - Adaptable panels and customizable workflows
7. Survivability - Robust error handling and graceful degradation
8. Economy - Efficient resource usage and minimal overhead

This controller serves as the master interface for:
- BTC Processor control and monitoring
- Trading visualization and management
- System resource oversight
- Mathematical engine coordination
- Custom panel installation and management
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import websockets
import numpy as np

# Core system imports
from .btc_processor_controller import BTCProcessorController
from .visual_integration_bridge import VisualIntegrationBridge, VisualMetrics, PatternState
from .ui_integration_bridge import UIIntegrationBridge, SystemMetrics, TradingMetrics
from .sustainment_underlay_controller import SustainmentUnderlayController

logger = logging.getLogger(__name__)

class PanelType(Enum):
    """Types of visual panels available"""
    TRADING = "trading"
    SYSTEM_MONITOR = "system_monitor" 
    BTC_PROCESSOR = "btc_processor"
    MATH_ENGINE = "math_engine"
    CUSTOM = "custom"
    ANALYSIS = "analysis"
    CONTROLS = "controls"
    VISUALIZATION = "visualization"

class ControlMode(Enum):
    """Operating modes for the visual controller"""
    LIVE_TRADING = "live_trading"
    TESTING = "testing"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    EMERGENCY = "emergency"

@dataclass
class PanelConfiguration:
    """Configuration for a visual panel"""
    panel_id: str
    panel_type: PanelType
    title: str
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    size: Dict[str, int] = field(default_factory=lambda: {"width": 400, "height": 300})
    visible: bool = True
    resizable: bool = True
    closable: bool = True
    customizable: bool = True
    update_frequency: float = 1.0  # seconds
    
@dataclass
class ToggleControl:
    """Simple toggle control for features"""
    control_id: str
    label: str
    description: str
    enabled: bool = True
    category: str = "general"
    requires_confirmation: bool = False
    impact_level: str = "low"  # low, medium, high, critical
    
@dataclass
class SliderControl:
    """Slider control for continuous values"""
    control_id: str
    label: str
    description: str
    current_value: float
    min_value: float
    max_value: float
    step: float = 0.1
    unit: str = ""
    category: str = "general"
    
@dataclass
class VisualState:
    """Complete state of the visual interface"""
    mode: ControlMode
    active_panels: List[str]
    panel_configurations: Dict[str, PanelConfiguration]
    toggle_states: Dict[str, bool]
    slider_values: Dict[str, float]
    last_update: datetime
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class UnifiedVisualController:
    """
    Master visual controller implementing sustainment principles.
    
    Provides a unified interface for all Schwabot systems with emphasis on:
    - Simple, non-intrusive controls
    - Real-time visual feedback
    - Customizable panel layouts
    - Seamless feature toggling
    - Emergency procedures
    """
    
    def __init__(self, 
                 btc_controller: BTCProcessorController,
                 visual_bridge: VisualIntegrationBridge,
                 ui_bridge: UIIntegrationBridge,
                 sustainment_controller: SustainmentUnderlayController,
                 websocket_port: int = 8080):
        
        # Core controllers
        self.btc_controller = btc_controller
        self.visual_bridge = visual_bridge
        self.ui_bridge = ui_bridge
        self.sustainment_controller = sustainment_controller
        
        # WebSocket server for real-time UI updates
        self.websocket_port = websocket_port
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.websocket_server = None
        
        # Visual state management
        self.visual_state = VisualState(
            mode=ControlMode.DEVELOPMENT,
            active_panels=[],
            panel_configurations={},
            toggle_states={},
            slider_values={},
            last_update=datetime.now()
        )
        
        # Control definitions
        self.toggle_controls: Dict[str, ToggleControl] = {}
        self.slider_controls: Dict[str, SliderControl] = {}
        self.panel_registry: Dict[str, PanelConfiguration] = {}
        
        # Real-time data streams
        self.live_data_streams: Dict[str, List[Dict]] = {
            "system_metrics": [],
            "btc_processor": [],
            "trading_data": [],
            "math_engine": [],
            "error_log": []
        }
        
        # Threading and control
        self.controller_active = False
        self.update_thread = None
        self.websocket_thread = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.update_count = 0
        self.ui_updates_sent = 0
        self.start_time = datetime.now()
        
        # Initialize the interface
        self._initialize_controls()
        self._initialize_panels()
        
        logger.info("Unified Visual Controller initialized")

    def _initialize_controls(self) -> None:
        """Initialize all toggle and slider controls based on sustainment principles"""
        
        # BTC Processor Controls (Simplicity + Economy principles)
        self.toggle_controls.update({
            "btc_mining_analysis": ToggleControl(
                "btc_mining_analysis", "Mining Analysis", 
                "Advanced mining pattern analysis", True, "btc_processor", False, "medium"
            ),
            "btc_hash_generation": ToggleControl(
                "btc_hash_generation", "Hash Generation",
                "Core hash generation functionality", True, "btc_processor", True, "critical"
            ),
            "btc_memory_management": ToggleControl(
                "btc_memory_management", "Memory Management",
                "Automatic memory optimization", True, "btc_processor", False, "high"
            ),
            "btc_load_balancing": ToggleControl(
                "btc_load_balancing", "Load Balancing",
                "CPU/GPU load distribution", True, "btc_processor", False, "medium"
            ),
            "btc_backlog_processing": ToggleControl(
                "btc_backlog_processing", "Backlog Processing",
                "Process queued analysis tasks", True, "btc_processor", False, "low"
            ),
        })
        
        # Trading Controls (Responsiveness + Integration principles)
        self.toggle_controls.update({
            "live_trading": ToggleControl(
                "live_trading", "Live Trading",
                "Execute live trades", False, "trading", True, "critical"
            ),
            "pattern_recognition": ToggleControl(
                "pattern_recognition", "Pattern Recognition",
                "Real-time pattern analysis", True, "trading", False, "high"
            ),
            "risk_monitoring": ToggleControl(
                "risk_monitoring", "Risk Monitoring", 
                "Continuous risk assessment", True, "trading", False, "high"
            ),
            "auto_rebalancing": ToggleControl(
                "auto_rebalancing", "Auto Rebalancing",
                "Automatic portfolio rebalancing", False, "trading", False, "medium"
            ),
        })
        
        # System Controls (Survivability + Continuity principles)
        self.toggle_controls.update({
            "emergency_mode": ToggleControl(
                "emergency_mode", "Emergency Mode",
                "Emergency system protection", False, "system", True, "critical"
            ),
            "auto_resource_management": ToggleControl(
                "auto_resource_management", "Auto Resource Management",
                "Automatic resource optimization", True, "system", False, "high"
            ),
            "thermal_protection": ToggleControl(
                "thermal_protection", "Thermal Protection",
                "Hardware thermal monitoring", True, "system", False, "high"
            ),
            "performance_logging": ToggleControl(
                "performance_logging", "Performance Logging",
                "Detailed performance metrics", True, "system", False, "low"
            ),
        })
        
        # Mathematical Engine Controls (Anticipation + Improvisation principles)
        self.toggle_controls.update({
            "math_v3_engine": ToggleControl(
                "math_v3_engine", "Math v3 Engine",
                "Advanced mathematical processing", True, "math", False, "high"
            ),
            "fractal_analysis": ToggleControl(
                "fractal_analysis", "Fractal Analysis",
                "Fractal pattern processing", True, "math", False, "medium"
            ),
            "quantum_processing": ToggleControl(
                "quantum_processing", "Quantum Processing",
                "Quantum algorithm simulation", False, "math", False, "low"
            ),
        })
        
        # Slider Controls for fine-tuning
        self.slider_controls.update({
            "max_memory_usage": SliderControl(
                "max_memory_usage", "Max Memory Usage", 
                "Maximum memory allocation", 10.0, 1.0, 50.0, 0.5, "GB", "resources"
            ),
            "cpu_usage_limit": SliderControl(
                "cpu_usage_limit", "CPU Usage Limit",
                "Maximum CPU utilization", 80.0, 10.0, 100.0, 5.0, "%", "resources"
            ),
            "gpu_usage_limit": SliderControl(
                "gpu_usage_limit", "GPU Usage Limit", 
                "Maximum GPU utilization", 85.0, 10.0, 100.0, 5.0, "%", "resources"
            ),
            "update_frequency": SliderControl(
                "update_frequency", "Update Frequency",
                "UI update rate", 1.0, 0.1, 10.0, 0.1, "Hz", "interface"
            ),
            "risk_tolerance": SliderControl(
                "risk_tolerance", "Risk Tolerance",
                "Trading risk level", 0.5, 0.0, 1.0, 0.05, "", "trading"
            ),
        })
        
        # Initialize states from current controllers
        self._sync_control_states()

    def _initialize_panels(self) -> None:
        """Initialize default panel configurations"""
        
        # BTC Processor Panel (Primary focus)
        self.panel_registry["btc_processor"] = PanelConfiguration(
            panel_id="btc_processor",
            panel_type=PanelType.BTC_PROCESSOR,
            title="BTC Processor Control",
            position={"x": 10, "y": 60},
            size={"width": 450, "height": 400},
            update_frequency=0.5
        )
        
        # Trading Overview Panel
        self.panel_registry["trading_overview"] = PanelConfiguration(
            panel_id="trading_overview", 
            panel_type=PanelType.TRADING,
            title="Trading Overview",
            position={"x": 470, "y": 60},
            size={"width": 400, "height": 300},
            update_frequency=1.0
        )
        
        # System Monitor Panel
        self.panel_registry["system_monitor"] = PanelConfiguration(
            panel_id="system_monitor",
            panel_type=PanelType.SYSTEM_MONITOR,
            title="System Resources",
            position={"x": 880, "y": 60},
            size={"width": 350, "height": 250},
            update_frequency=2.0
        )
        
        # Quick Controls Panel (Innovation - always accessible)
        self.panel_registry["quick_controls"] = PanelConfiguration(
            panel_id="quick_controls",
            panel_type=PanelType.CONTROLS,
            title="Quick Controls",
            position={"x": 10, "y": 470},
            size={"width": 860, "height": 120},
            update_frequency=5.0
        )
        
        # Mathematical Engine Panel
        self.panel_registry["math_engine"] = PanelConfiguration(
            panel_id="math_engine",
            panel_type=PanelType.MATH_ENGINE,
            title="Mathematical Engine",
            position={"x": 880, "y": 320},
            size={"width": 350, "height": 270},
            update_frequency=1.0
        )
        
        # Analysis Panel (for deep dive when needed)
        self.panel_registry["analysis"] = PanelConfiguration(
            panel_id="analysis",
            panel_type=PanelType.ANALYSIS,
            title="Advanced Analysis",
            position={"x": 470, "y": 370},
            size={"width": 400, "height": 220},
            visible=False,  # Hidden by default
            update_frequency=3.0
        )
        
        # Custom Visualization Panel (Improvisation principle)
        self.panel_registry["custom_viz"] = PanelConfiguration(
            panel_id="custom_viz",
            panel_type=PanelType.VISUALIZATION,
            title="Custom Visualizations",
            position={"x": 1240, "y": 60},
            size={"width": 300, "height": 530},
            visible=False,  # Show when needed
            update_frequency=2.0
        )

    def _sync_control_states(self) -> None:
        """Synchronize control states with underlying systems"""
        
        # Get current BTC processor state
        btc_status = self.btc_controller.get_current_status()
        
        # Update toggle states
        if 'config' in btc_status:
            config = btc_status['config']
            self.visual_state.toggle_states.update({
                "btc_mining_analysis": config.get('mining_analysis_enabled', True),
                "btc_hash_generation": config.get('hash_generation_enabled', True),
                "btc_memory_management": config.get('memory_management_enabled', True),
                "btc_load_balancing": config.get('load_balancing_enabled', True),
                "btc_backlog_processing": config.get('backlog_processing', True),
            })
        
        # Update slider values
        if 'thresholds' in btc_status:
            thresholds = btc_status['thresholds']
            self.visual_state.slider_values.update({
                "max_memory_usage": thresholds.get('memory_critical_gb', 10.0),
                "cpu_usage_limit": thresholds.get('cpu_critical_percent', 80.0),
                "gpu_usage_limit": thresholds.get('gpu_critical_percent', 85.0),
            })

    async def start_visual_controller(self) -> None:
        """Start the unified visual controller"""
        
        if self.controller_active:
            logger.warning("Visual controller already active")
            return
        
        self.controller_active = True
        
        # Start WebSocket server for real-time UI communication
        await self._start_websocket_server()
        
        # Start update threads
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start subsidiary controllers
        await self.btc_controller.start_monitoring()
        self.visual_bridge.start_visual_bridge()
        self.ui_bridge.start()
        
        logger.info("Unified Visual Controller started successfully")

    async def stop_visual_controller(self) -> None:
        """Stop the visual controller gracefully"""
        
        self.controller_active = False
        
        # Stop subsidiary controllers
        await self.btc_controller.stop_monitoring()
        self.visual_bridge.stop_visual_bridge()
        self.ui_bridge.stop()
        
        # Stop WebSocket server
        if self.websocket_server:
            await self._close_websocket_clients()
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Wait for update thread
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("Unified Visual Controller stopped")

    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time UI updates"""
        
        async def handle_client(websocket, path):
            """Handle WebSocket client connections"""
            logger.info(f"Visual controller client connected: {websocket.remote_address}")
            self.websocket_clients.add(websocket)
            
            try:
                # Send initial state
                await self._send_initial_state(websocket)
                
                # Handle incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_client_message(websocket, data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from client: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("Visual controller client disconnected")
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
        
        try:
            self.websocket_server = await websockets.serve(
                handle_client, "localhost", self.websocket_port
            )
            logger.info(f"Visual controller WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    async def _send_initial_state(self, websocket) -> None:
        """Send initial visual state to a new client"""
        
        initial_state = {
            "type": "initial_state",
            "data": {
                "visual_state": asdict(self.visual_state),
                "toggle_controls": {k: asdict(v) for k, v in self.toggle_controls.items()},
                "slider_controls": {k: asdict(v) for k, v in self.slider_controls.items()},
                "panel_registry": {k: asdict(v) for k, v in self.panel_registry.items()},
                "current_mode": self.visual_state.mode.value,
                "available_modes": [mode.value for mode in ControlMode],
            }
        }
        
        await websocket.send(json.dumps(initial_state, default=str))

    async def _handle_client_message(self, websocket, data: Dict[str, Any]) -> None:
        """Handle messages from WebSocket clients"""
        
        message_type = data.get("type")
        payload = data.get("data", {})
        
        try:
            if message_type == "toggle_control":
                await self._handle_toggle_control(payload)
            elif message_type == "slider_control":
                await self._handle_slider_control(payload)
            elif message_type == "panel_action":
                await self._handle_panel_action(payload)
            elif message_type == "mode_change":
                await self._handle_mode_change(payload)
            elif message_type == "emergency_stop":
                await self._handle_emergency_stop(payload)
            elif message_type == "install_panel":
                await self._handle_install_panel(payload)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
            # Send acknowledgment
            await websocket.send(json.dumps({
                "type": "ack",
                "message_type": message_type,
                "success": True
            }))
            
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def _handle_toggle_control(self, payload: Dict[str, Any]) -> None:
        """Handle toggle control changes"""
        
        control_id = payload.get("control_id")
        new_state = payload.get("enabled")
        
        if control_id not in self.toggle_controls:
            raise ValueError(f"Unknown toggle control: {control_id}")
        
        control = self.toggle_controls[control_id]
        
        # Check if confirmation is required
        if control.requires_confirmation and not payload.get("confirmed"):
            # Send confirmation request
            await self._broadcast_to_clients({
                "type": "confirmation_required",
                "data": {
                    "control_id": control_id,
                    "action": "toggle",
                    "new_state": new_state,
                    "warning": f"This action will affect {control.description.lower()}"
                }
            })
            return
        
        # Apply the control change
        if control.category == "btc_processor":
            await self._apply_btc_control(control_id, new_state)
        elif control.category == "trading":
            await self._apply_trading_control(control_id, new_state)
        elif control.category == "system":
            await self._apply_system_control(control_id, new_state)
        elif control.category == "math":
            await self._apply_math_control(control_id, new_state)
        
        # Update visual state
        self.visual_state.toggle_states[control_id] = new_state
        
        # Broadcast update to all clients
        await self._broadcast_control_update("toggle", control_id, new_state)

    async def _apply_btc_control(self, control_id: str, enabled: bool) -> None:
        """Apply BTC processor control changes"""
        
        feature_map = {
            "btc_mining_analysis": "mining_analysis",
            "btc_hash_generation": "hash_generation", 
            "btc_memory_management": "memory_management",
            "btc_load_balancing": "load_balancing",
            "btc_backlog_processing": "backlog_processing"
        }
        
        feature_name = feature_map.get(control_id)
        if feature_name:
            if enabled:
                await self.btc_controller.enable_feature(feature_name)
            else:
                await self.btc_controller.disable_feature(feature_name)

    async def _apply_trading_control(self, control_id: str, enabled: bool) -> None:
        """Apply trading control changes"""
        
        # Route to appropriate trading controller
        if control_id == "live_trading":
            # Critical control - requires special handling
            if enabled:
                await self._enable_live_trading()
            else:
                await self._disable_live_trading()

    async def _apply_system_control(self, control_id: str, enabled: bool) -> None:
        """Apply system control changes"""
        
        if control_id == "emergency_mode":
            if enabled:
                await self._enter_emergency_mode()
            else:
                await self._exit_emergency_mode()

    async def _apply_math_control(self, control_id: str, enabled: bool) -> None:
        """Apply mathematical engine control changes"""
        
        # Route to math engine controller
        pass  # Implementation depends on specific math engines

    def _update_loop(self) -> None:
        """Main update loop for gathering and broadcasting visual data"""
        
        while self.controller_active:
            try:
                start_time = time.time()
                
                # Gather data from all controllers
                self._gather_visual_data()
                
                # Update panel data streams
                self._update_data_streams()
                
                # Broadcast updates to clients
                asyncio.run(self._broadcast_periodic_updates())
                
                # Performance tracking
                self.update_count += 1
                update_time = time.time() - start_time
                self.visual_state.performance_metrics["update_time_ms"] = update_time * 1000
                
                # Sleep based on update frequency
                sleep_time = max(0.1, 1.0 / self.visual_state.slider_values.get("update_frequency", 1.0))
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in visual controller update loop: {e}")
                self.visual_state.error_count += 1
                time.sleep(1.0)

    def _gather_visual_data(self) -> None:
        """Gather data from all subsystems for visual display"""
        
        # Get BTC processor status
        btc_status = self.btc_controller.get_current_status()
        
        # Get UI bridge data
        system_metrics = self.ui_bridge.get_current_system_metrics()
        trading_metrics = self.ui_bridge.get_current_trading_metrics()
        
        # Get visual bridge data
        if hasattr(self.visual_bridge, 'current_metrics'):
            visual_metrics = self.visual_bridge.current_metrics
        else:
            visual_metrics = None
        
        # Update live data streams
        timestamp = datetime.now()
        
        self.live_data_streams["btc_processor"].append({
            "timestamp": timestamp,
            "status": btc_status,
            "metrics": btc_status.get("metrics", {})
        })
        
        if system_metrics:
            self.live_data_streams["system_metrics"].append({
                "timestamp": timestamp,
                "metrics": asdict(system_metrics)
            })
        
        if trading_metrics:
            self.live_data_streams["trading_data"].append({
                "timestamp": timestamp,
                "metrics": asdict(trading_metrics)
            })
        
        # Trim old data to prevent memory bloat (Economy principle)
        max_history = 1000
        for stream in self.live_data_streams.values():
            if len(stream) > max_history:
                stream[:] = stream[-max_history:]

    def _update_data_streams(self) -> None:
        """Update data streams for each panel type"""
        
        # Update each panel's data based on its requirements
        for panel_id, panel_config in self.panel_registry.items():
            if not panel_config.visible:
                continue
                
            panel_data = self._generate_panel_data(panel_config)
            
            # Store panel-specific data
            if "panel_data" not in self.live_data_streams:
                self.live_data_streams["panel_data"] = {}
            
            self.live_data_streams["panel_data"][panel_id] = panel_data

    def _generate_panel_data(self, panel_config: PanelConfiguration) -> Dict[str, Any]:
        """Generate data for a specific panel"""
        
        if panel_config.panel_type == PanelType.BTC_PROCESSOR:
            return self._generate_btc_processor_panel_data()
        elif panel_config.panel_type == PanelType.TRADING:
            return self._generate_trading_panel_data()
        elif panel_config.panel_type == PanelType.SYSTEM_MONITOR:
            return self._generate_system_monitor_panel_data()
        elif panel_config.panel_type == PanelType.CONTROLS:
            return self._generate_controls_panel_data()
        elif panel_config.panel_type == PanelType.MATH_ENGINE:
            return self._generate_math_engine_panel_data()
        else:
            return {"type": "unknown", "data": {}}

    def _generate_btc_processor_panel_data(self) -> Dict[str, Any]:
        """Generate BTC processor panel data"""
        
        btc_status = self.btc_controller.get_current_status()
        
        return {
            "type": "btc_processor",
            "status": btc_status.get("status", "unknown"),
            "metrics": btc_status.get("metrics", {}),
            "config": btc_status.get("config", {}),
            "thresholds": btc_status.get("thresholds", {}),
            "feature_states": {
                control_id: self.visual_state.toggle_states.get(control_id, False)
                for control_id in ["btc_mining_analysis", "btc_hash_generation", 
                                 "btc_memory_management", "btc_load_balancing", 
                                 "btc_backlog_processing"]
            },
            "resource_usage": {
                "memory": btc_status.get("metrics", {}).get("memory_usage_gb", 0),
                "cpu": btc_status.get("metrics", {}).get("cpu_usage", 0),
                "gpu": btc_status.get("metrics", {}).get("gpu_usage", 0)
            }
        }

    def _generate_trading_panel_data(self) -> Dict[str, Any]:
        """Generate trading panel data"""
        
        trading_metrics = self.ui_bridge.get_current_trading_metrics()
        
        if trading_metrics:
            return {
                "type": "trading",
                "metrics": asdict(trading_metrics),
                "live_trading_enabled": self.visual_state.toggle_states.get("live_trading", False),
                "recent_decisions": self.ui_bridge.get_recent_decisions(5)
            }
        else:
            return {"type": "trading", "metrics": {}, "status": "no_data"}

    def _generate_system_monitor_panel_data(self) -> Dict[str, Any]:
        """Generate system monitor panel data"""
        
        system_metrics = self.ui_bridge.get_current_system_metrics()
        
        if system_metrics:
            return {
                "type": "system_monitor",
                "metrics": asdict(system_metrics),
                "emergency_mode": self.visual_state.toggle_states.get("emergency_mode", False),
                "thermal_protection": self.visual_state.toggle_states.get("thermal_protection", True),
                "resource_limits": {
                    "memory": self.visual_state.slider_values.get("max_memory_usage", 10.0),
                    "cpu": self.visual_state.slider_values.get("cpu_usage_limit", 80.0),
                    "gpu": self.visual_state.slider_values.get("gpu_usage_limit", 85.0)
                }
            }
        else:
            return {"type": "system_monitor", "metrics": {}, "status": "no_data"}

    def _generate_controls_panel_data(self) -> Dict[str, Any]:
        """Generate quick controls panel data"""
        
        return {
            "type": "controls",
            "critical_toggles": {
                "live_trading": self.visual_state.toggle_states.get("live_trading", False),
                "emergency_mode": self.visual_state.toggle_states.get("emergency_mode", False),
                "btc_hash_generation": self.visual_state.toggle_states.get("btc_hash_generation", True)
            },
            "key_sliders": {
                "risk_tolerance": self.visual_state.slider_values.get("risk_tolerance", 0.5),
                "max_memory_usage": self.visual_state.slider_values.get("max_memory_usage", 10.0)
            },
            "current_mode": self.visual_state.mode.value,
            "system_health": self._calculate_overall_health()
        }

    def _generate_math_engine_panel_data(self) -> Dict[str, Any]:
        """Generate mathematical engine panel data"""
        
        return {
            "type": "math_engine",
            "engine_states": {
                "math_v3_engine": self.visual_state.toggle_states.get("math_v3_engine", True),
                "fractal_analysis": self.visual_state.toggle_states.get("fractal_analysis", True),
                "quantum_processing": self.visual_state.toggle_states.get("quantum_processing", False)
            },
            "performance_metrics": self.visual_state.performance_metrics,
            "sustainment_principles": self._get_sustainment_metrics()
        }

    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        
        health_factors = []
        
        # BTC processor health
        btc_status = self.btc_controller.get_current_status()
        if btc_status.get("status") == "healthy":
            health_factors.append(1.0)
        elif btc_status.get("status") == "warning":
            health_factors.append(0.7)
        else:
            health_factors.append(0.3)
        
        # System metrics health
        system_metrics = self.ui_bridge.get_current_system_metrics()
        if system_metrics:
            health_factors.append(system_metrics.system_health)
        
        # Error rate health
        error_rate = self.visual_state.error_count / max(1, self.update_count)
        error_health = max(0.0, 1.0 - error_rate * 10)
        health_factors.append(error_health)
        
        return sum(health_factors) / len(health_factors) if health_factors else 0.5

    def _get_sustainment_metrics(self) -> Dict[str, float]:
        """Get sustainment principle metrics"""
        
        # This would integrate with the sustainment underlay controller
        return {
            "anticipation": 0.85,
            "continuity": 0.92,
            "responsiveness": 0.88,
            "integration": 0.90,
            "simplicity": 0.95,
            "improvisation": 0.80,
            "survivability": 0.87,
            "economy": 0.91
        }

    async def _broadcast_periodic_updates(self) -> None:
        """Broadcast periodic updates to all connected clients"""
        
        if not self.websocket_clients:
            return
        
        update_data = {
            "type": "periodic_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "visual_state": asdict(self.visual_state),
                "panel_data": self.live_data_streams.get("panel_data", {}),
                "performance": self.visual_state.performance_metrics
            }
        }
        
        await self._broadcast_to_clients(update_data)

    async def _broadcast_control_update(self, control_type: str, control_id: str, new_value: Any) -> None:
        """Broadcast control state changes to all clients"""
        
        update_data = {
            "type": "control_update",
            "control_type": control_type,
            "control_id": control_id,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_to_clients(update_data)

    async def _broadcast_to_clients(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        message = json.dumps(data, default=str)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
                self.ui_updates_sent += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients

    async def _close_websocket_clients(self) -> None:
        """Close all WebSocket client connections"""
        
        for client in self.websocket_clients.copy():
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket client: {e}")
        
        self.websocket_clients.clear()

    # Mode management methods
    async def switch_to_live_trading_mode(self) -> None:
        """Switch to live trading mode with appropriate safety checks"""
        
        # Ensure BTC processor is in optimal state for live trading
        await self.btc_controller.disable_all_analysis_features()
        
        # Set conservative resource limits
        self.visual_state.slider_values.update({
            "max_memory_usage": 8.0,
            "cpu_usage_limit": 70.0,
            "gpu_usage_limit": 80.0
        })
        
        self.visual_state.mode = ControlMode.LIVE_TRADING
        await self._broadcast_mode_change()

    async def switch_to_testing_mode(self) -> None:
        """Switch to testing mode for safe experimentation"""
        
        # Disable live trading
        self.visual_state.toggle_states["live_trading"] = False
        
        # Enable analysis features for testing
        await self.btc_controller.enable_all_analysis_features()
        
        self.visual_state.mode = ControlMode.TESTING
        await self._broadcast_mode_change()

    async def _enter_emergency_mode(self) -> None:
        """Enter emergency mode with immediate protective actions"""
        
        logger.warning("Entering emergency mode")
        
        # Disable live trading immediately
        self.visual_state.toggle_states["live_trading"] = False
        
        # Trigger emergency cleanup in BTC processor
        await self.btc_controller._emergency_memory_cleanup()
        
        # Set ultra-conservative limits
        self.visual_state.slider_values.update({
            "max_memory_usage": 4.0,
            "cpu_usage_limit": 50.0,
            "gpu_usage_limit": 60.0
        })
        
        self.visual_state.mode = ControlMode.EMERGENCY
        await self._broadcast_mode_change()

    async def _exit_emergency_mode(self) -> None:
        """Exit emergency mode and restore normal operation"""
        
        logger.info("Exiting emergency mode")
        
        # Restore normal limits
        self.visual_state.slider_values.update({
            "max_memory_usage": 10.0,
            "cpu_usage_limit": 80.0,
            "gpu_usage_limit": 85.0
        })
        
        self.visual_state.mode = ControlMode.DEVELOPMENT
        await self._broadcast_mode_change()

    async def _broadcast_mode_change(self) -> None:
        """Broadcast mode change to all clients"""
        
        await self._broadcast_to_clients({
            "type": "mode_change",
            "new_mode": self.visual_state.mode.value,
            "timestamp": datetime.now().isoformat(),
            "updated_controls": {
                "toggles": self.visual_state.toggle_states,
                "sliders": self.visual_state.slider_values
            }
        })

    # Panel management methods (Improvisation principle)
    async def install_custom_panel(self, panel_config: Dict[str, Any]) -> str:
        """Install a custom panel into the interface"""
        
        panel_id = panel_config.get("panel_id", f"custom_{len(self.panel_registry)}")
        
        # Validate panel configuration
        required_fields = ["title", "panel_type"]
        for field in required_fields:
            if field not in panel_config:
                raise ValueError(f"Missing required field: {field}")
        
        # Create panel configuration
        new_panel = PanelConfiguration(
            panel_id=panel_id,
            panel_type=PanelType.CUSTOM,
            title=panel_config["title"],
            position=panel_config.get("position", {"x": 100, "y": 100}),
            size=panel_config.get("size", {"width": 300, "height": 200}),
            visible=panel_config.get("visible", True),
            update_frequency=panel_config.get("update_frequency", 2.0)
        )
        
        # Register the panel
        self.panel_registry[panel_id] = new_panel
        
        # Broadcast panel installation
        await self._broadcast_to_clients({
            "type": "panel_installed",
            "panel_id": panel_id,
            "panel_config": asdict(new_panel)
        })
        
        logger.info(f"Custom panel installed: {panel_id}")
        return panel_id

    def get_visual_state(self) -> Dict[str, Any]:
        """Get current visual state for external access"""
        
        return {
            "mode": self.visual_state.mode.value,
            "active_panels": list(self.panel_registry.keys()),
            "toggle_states": self.visual_state.toggle_states,
            "slider_values": self.visual_state.slider_values,
            "performance_metrics": self.visual_state.performance_metrics,
            "error_count": self.visual_state.error_count,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }

def create_unified_visual_controller(
    btc_controller: BTCProcessorController,
    visual_bridge: VisualIntegrationBridge,
    ui_bridge: UIIntegrationBridge,
    sustainment_controller: SustainmentUnderlayController,
    websocket_port: int = 8080
) -> UnifiedVisualController:
    """Factory function to create a unified visual controller"""
    
    return UnifiedVisualController(
        btc_controller=btc_controller,
        visual_bridge=visual_bridge,
        ui_bridge=ui_bridge,
        sustainment_controller=sustainment_controller,
        websocket_port=websocket_port
    ) 