"""
UI State Bridge
===============

Bridges all Schwabot mathematical systems into a unified UI state that can be
consumed by any frontend (React, Dear PyGui, web dashboard, etc).

This is the critical layer that translates complex mathematical states into
clean, actionable UI data structures.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import numpy as np

# Core system imports
from .sustainment_underlay_controller import SustainmentUnderlayController, SustainmentPrinciple
from .thermal_zone_manager import ThermalZoneManager, ThermalZone
from .profit_navigator import AntiPoleProfitNavigator
from .fractal_core import FractalCore
from .collapse_confidence import CollapseConfidenceEngine

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """Overall system status levels"""
    OPTIMAL = "optimal"           # All systems green
    OPERATIONAL = "operational"   # Minor warnings
    DEGRADED = "degraded"        # Performance issues
    CRITICAL = "critical"        # Major problems
    OFFLINE = "offline"          # System down

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class UIAlert:
    """User-facing alert"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    actionable: bool = False
    action_text: Optional[str] = None
    action_callback: Optional[str] = None

@dataclass
class UISystemHealth:
    """Overall system health for dashboard"""
    status: SystemStatus
    sustainment_index: float
    health_score: float
    uptime: timedelta
    active_alerts: int
    critical_alerts: int
    
    # Key metrics for quick dashboard view
    profit_24h: float
    trades_today: int
    success_rate: float
    thermal_status: str
    gpu_utilization: float
    
    # Status indicators for major subsystems
    data_feed_status: str
    trading_engine_status: str
    risk_management_status: str
    hardware_status: str

@dataclass
class UISustainmentRadar:
    """8-principle sustainment radar for dashboard"""
    anticipation: float
    integration: float  
    responsiveness: float
    simplicity: float
    economy: float
    survivability: float
    continuity: float
    improvisation: float
    
    # Metadata
    overall_index: float
    critical_threshold: float
    violations: Dict[str, int]
    last_correction: Optional[str]

@dataclass
class UIHardwareState:
    """Hardware monitoring state"""
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    cpu_temp: float
    gpu_temp: float
    thermal_zone: str
    power_usage: float
    
    # Status flags
    thermal_throttling: bool
    gpu_available: bool
    failover_active: bool
    last_handoff: Optional[str]

@dataclass
class UITradingState:
    """Current trading state and performance"""
    # Position information
    current_positions: Dict[str, float]
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Performance metrics
    daily_return: float
    weekly_return: float
    monthly_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    
    # Active trading info
    active_strategies: List[str]
    pending_orders: int
    last_trade_time: Optional[datetime]
    strategy_performance: Dict[str, Dict[str, float]]

@dataclass
class UIVisualizationData:
    """Data for the Tesseract and other visualizers"""
    # Fractal state
    fractal_coherence: float
    fractal_entropy: float
    fractal_phase: float
    pattern_strength: float
    
    # Market data
    price_data: List[Dict[str, Any]]
    volume_data: List[Dict[str, Any]]
    volatility: float
    
    # Visualization elements
    active_overlays: List[str]
    tesseract_frame_id: str
    glyph_count: int
    profit_tier: str

class UIStateBridge:
    """
    Central bridge that aggregates all system states into clean UI data structures.
    
    This runs continuously and maintains the current UI state that any frontend
    can consume via get_ui_state() or through WebSocket streaming.
    """
    
    def __init__(self,
                 sustainment_controller: SustainmentUnderlayController,
                 thermal_manager: ThermalZoneManager,
                 profit_navigator: AntiPoleProfitNavigator,
                 fractal_core: FractalCore,
                 confidence_engine: Optional[CollapseConfidenceEngine] = None):
        """
        Initialize UI state bridge with all core controllers
        
        Args:
            sustainment_controller: Main sustainment underlay
            thermal_manager: Thermal zone management
            profit_navigator: Profit and trading logic
            fractal_core: Fractal pattern processing
            confidence_engine: Optional confidence scoring
        """
        
        # Core controllers
        self.sustainment_controller = sustainment_controller
        self.thermal_manager = thermal_manager
        self.profit_navigator = profit_navigator
        self.fractal_core = fractal_core
        self.confidence_engine = confidence_engine
        
        # UI state tracking
        self.current_ui_state: Dict[str, Any] = {}
        self.ui_alerts: deque = deque(maxlen=100)
        self.state_history: deque = deque(maxlen=1000)
        
        # WebSocket connections for real-time updates
        self.websocket_clients: List[Any] = []
        self.update_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitoring_active = False
        self.update_thread = None
        self.last_update = datetime.now()
        self._lock = threading.RLock()
        
        # Performance tracking
        self.update_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
        logger.info("UI State Bridge initialized - Ready to serve frontend data")

    def start_ui_monitoring(self, update_interval: float = 1.0) -> None:
        """Start continuous UI state monitoring and updates"""
        if self.monitoring_active:
            logger.warning("UI monitoring already active")
            return
        
        self.monitoring_active = True
        self.update_thread = threading.Thread(
            target=self._continuous_update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"UI monitoring started (interval: {update_interval}s)")

    def stop_ui_monitoring(self) -> None:
        """Stop UI monitoring"""
        self.monitoring_active = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        logger.info("UI monitoring stopped")

    def _continuous_update_loop(self, interval: float) -> None:
        """Continuous update loop that refreshes UI state"""
        while self.monitoring_active:
            try:
                self.update_ui_state()
                time.sleep(interval)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in UI update loop: {e}")
                time.sleep(interval)

    def update_ui_state(self) -> Dict[str, Any]:
        """Update complete UI state from all controllers"""
        with self._lock:
            try:
                # Get system health overview
                system_health = self._build_system_health()
                
                # Get sustainment radar data
                sustainment_radar = self._build_sustainment_radar()
                
                # Get hardware state
                hardware_state = self._build_hardware_state()
                
                # Get trading state
                trading_state = self._build_trading_state()
                
                # Get visualization data
                visualization_data = self._build_visualization_data()
                
                # Get recent alerts
                recent_alerts = list(self.ui_alerts)[-10:]  # Last 10 alerts
                
                # Build complete UI state
                ui_state = {
                    'timestamp': datetime.now().isoformat(),
                    'system_health': asdict(system_health),
                    'sustainment_radar': asdict(sustainment_radar),
                    'hardware_state': asdict(hardware_state),
                    'trading_state': asdict(trading_state),
                    'visualization_data': asdict(visualization_data),
                    'alerts': [asdict(alert) for alert in recent_alerts],
                    'metadata': {
                        'update_count': self.update_count,
                        'error_count': self.error_count,
                        'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                        'last_update': self.last_update.isoformat()
                    }
                }
                
                # Update state and notify clients
                self.current_ui_state = ui_state
                self.state_history.append(ui_state)
                self.update_count += 1
                self.last_update = datetime.now()
                
                # Notify all registered callbacks
                self._notify_ui_callbacks(ui_state)
                
                return ui_state
                
            except Exception as e:
                logger.error(f"Failed to update UI state: {e}")
                return self.current_ui_state or {}

    def _build_system_health(self) -> UISystemHealth:
        """Build overall system health summary"""
        
        # Get sustainment status
        sustainment_status = self.sustainment_controller.get_sustainment_status()
        si = sustainment_status.get('sustainment_index', 0.5)
        health_score = sustainment_status.get('system_health_score', 0.5)
        
        # Determine overall status
        critical_alerts = len([a for a in self.ui_alerts if a.level == AlertLevel.CRITICAL])
        active_alerts = len(self.ui_alerts)
        
        if critical_alerts > 0:
            status = SystemStatus.CRITICAL
        elif si < 0.5:
            status = SystemStatus.DEGRADED
        elif active_alerts > 5:
            status = SystemStatus.OPERATIONAL
        else:
            status = SystemStatus.OPTIMAL
        
        # Get thermal status
        thermal_state = self.thermal_manager.get_current_state()
        thermal_status = thermal_state.zone.value if thermal_state else "unknown"
        gpu_util = thermal_state.load_gpu if thermal_state else 0.0
        
        # Get basic trading metrics (mock for now)
        profit_24h = 150.75  # Would come from profit navigator
        trades_today = 12
        success_rate = 0.67
        
        return UISystemHealth(
            status=status,
            sustainment_index=si,
            health_score=health_score,
            uptime=datetime.now() - self.start_time,
            active_alerts=active_alerts,
            critical_alerts=critical_alerts,
            profit_24h=profit_24h,
            trades_today=trades_today,
            success_rate=success_rate,
            thermal_status=thermal_status,
            gpu_utilization=gpu_util,
            data_feed_status="connected",
            trading_engine_status="active",
            risk_management_status="monitoring",
            hardware_status="nominal"
        )

    def _build_sustainment_radar(self) -> UISustainmentRadar:
        """Build sustainment radar data for 8-principle visualization"""
        
        sustainment_status = self.sustainment_controller.get_sustainment_status()
        
        if sustainment_status.get('status') == 'initializing':
            # Return default radar during initialization
            return UISustainmentRadar(
                anticipation=0.5, integration=0.5, responsiveness=0.5, simplicity=0.5,
                economy=0.5, survivability=0.5, continuity=0.5, improvisation=0.5,
                overall_index=0.5, critical_threshold=0.65, violations={}, last_correction=None
            )
        
        vector = sustainment_status['current_vector']
        
        return UISustainmentRadar(
            anticipation=vector['anticipation'],
            integration=vector['integration'],
            responsiveness=vector['responsiveness'],
            simplicity=vector['simplicity'],
            economy=vector['economy'],
            survivability=vector['survivability'],
            continuity=vector['continuity'],
            improvisation=vector['improvisation'],
            overall_index=sustainment_status['sustainment_index'],
            critical_threshold=sustainment_status['critical_threshold'],
            violations=sustainment_status['principle_violations'],
            last_correction=self._get_last_correction()
        )

    def _build_hardware_state(self) -> UIHardwareState:
        """Build hardware monitoring state"""
        
        thermal_state = self.thermal_manager.get_current_state()
        
        if thermal_state:
            return UIHardwareState(
                cpu_usage=thermal_state.load_cpu,
                gpu_usage=thermal_state.load_gpu,
                memory_usage=thermal_state.memory_usage,
                gpu_memory_usage=0.4,  # Mock for now
                cpu_temp=thermal_state.cpu_temp,
                gpu_temp=thermal_state.gpu_temp,
                thermal_zone=thermal_state.zone.value,
                power_usage=0.6,  # Mock
                thermal_throttling=thermal_state.zone == ThermalZone.HOT,
                gpu_available=True,
                failover_active=False,
                last_handoff=None
            )
        else:
            # Default/mock state
            return UIHardwareState(
                cpu_usage=0.3, gpu_usage=0.4, memory_usage=0.5, gpu_memory_usage=0.3,
                cpu_temp=55.0, gpu_temp=65.0, thermal_zone="normal", power_usage=0.5,
                thermal_throttling=False, gpu_available=True, failover_active=False,
                last_handoff=None
            )

    def _build_trading_state(self) -> UITradingState:
        """Build current trading state and performance"""
        
        # This would integrate with your profit navigator
        # For now, returning mock data structure
        
        return UITradingState(
            current_positions={"BTC": 0.5, "ETH": 2.1},
            total_equity=45678.32,
            unrealized_pnl=234.56,
            realized_pnl=1234.78,
            daily_return=0.023,
            weekly_return=0.067,
            monthly_return=0.145,
            max_drawdown=-0.08,
            sharpe_ratio=1.34,
            win_rate=0.67,
            active_strategies=["momentum", "reversal", "anti_pole"],
            pending_orders=3,
            last_trade_time=datetime.now() - timedelta(minutes=23),
            strategy_performance={
                "momentum": {"return": 0.045, "trades": 8, "win_rate": 0.75},
                "reversal": {"return": 0.012, "trades": 4, "win_rate": 0.50},
                "anti_pole": {"return": 0.089, "trades": 3, "win_rate": 1.0}
            }
        )

    def _build_visualization_data(self) -> UIVisualizationData:
        """Build data for Tesseract and other visualizers"""
        
        # Get fractal state if available
        fractal_state = None
        try:
            if hasattr(self.fractal_core, 'get_current_state'):
                fractal_state = self.fractal_core.get_current_state()
        except:
            pass
        
        if fractal_state:
            coherence = getattr(fractal_state, 'coherence', 0.6)
            entropy = getattr(fractal_state, 'entropy', 0.4)
            phase = getattr(fractal_state, 'phase', 0.0)
        else:
            coherence, entropy, phase = 0.6, 0.4, 0.0
        
        return UIVisualizationData(
            fractal_coherence=coherence,
            fractal_entropy=entropy,
            fractal_phase=phase,
            pattern_strength=0.75,
            price_data=[],  # Would be populated with recent market data
            volume_data=[],
            volatility=0.15,
            active_overlays=["fractal", "momentum", "thermal"],
            tesseract_frame_id=f"frame_{int(time.time())}",
            glyph_count=12,
            profit_tier="GOLD"
        )

    def _get_last_correction(self) -> Optional[str]:
        """Get description of last sustainment correction"""
        corrections = self.sustainment_controller.correction_history
        if corrections:
            last = corrections[-1]
            return f"{last.principle.value} -> {last.action_type}"
        return None

    def add_alert(self, level: AlertLevel, title: str, message: str, 
                  source: str, actionable: bool = False) -> str:
        """Add a new UI alert"""
        alert_id = f"alert_{int(time.time())}_{len(self.ui_alerts)}"
        
        alert = UIAlert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            actionable=actionable
        )
        
        self.ui_alerts.append(alert)
        logger.info(f"UI Alert: {level.value} - {title} from {source}")
        
        return alert_id

    def register_ui_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for UI state updates"""
        self.update_callbacks.append(callback)

    def _notify_ui_callbacks(self, ui_state: Dict[str, Any]) -> None:
        """Notify all registered UI callbacks of state update"""
        for callback in self.update_callbacks:
            try:
                callback(ui_state)
            except Exception as e:
                logger.error(f"Error in UI callback: {e}")

    def get_ui_state(self) -> Dict[str, Any]:
        """Get current UI state (thread-safe)"""
        with self._lock:
            return self.current_ui_state.copy() if self.current_ui_state else {}

    def get_ui_state_json(self) -> str:
        """Get current UI state as JSON string"""
        return json.dumps(self.get_ui_state(), default=str, indent=2)

    def export_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Export recent state history for debugging/analysis"""
        return list(self.state_history)[-limit:]

# Factory function for easy integration
def create_ui_bridge(sustainment_controller, thermal_manager, profit_navigator, fractal_core) -> UIStateBridge:
    """Factory function to create UI state bridge with all controllers"""
    
    bridge = UIStateBridge(
        sustainment_controller=sustainment_controller,
        thermal_manager=thermal_manager,
        profit_navigator=profit_navigator,
        fractal_core=fractal_core
    )
    
    # Start monitoring
    bridge.start_ui_monitoring()
    
    # Add initial system startup alert
    bridge.add_alert(
        AlertLevel.INFO,
        "System Initialized",
        "Schwabot UI State Bridge is now active and monitoring all systems",
        "ui_bridge"
    )
    
    logger.info("UI State Bridge created and monitoring started")
    return bridge 