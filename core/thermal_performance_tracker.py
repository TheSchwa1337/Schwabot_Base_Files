"""
Thermal Performance Tracker
===========================

Comprehensive thermal and performance tracking system that integrates with the
visual controller to provide real-time monitoring, tick analysis, and hover-based
information portals. This system tracks CPU/GPU usage, thermal states, trading
decisions, and provides detailed analytics for both backtesting and live trading.

Features:
- Real-time thermal and performance monitoring
- Tick-by-tick analysis with hover information
- CPU/GPU usage portioning and allocation tracking
- Trading decision correlation with thermal states
- Historical pattern analysis and visualization
- Cross-platform UI integration (Windows, Mac, Linux)
- Live runtime environment visualization
"""

import time
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque, defaultdict
import threading
import asyncio
from pathlib import Path

# Core system imports
from .thermal_zone_manager_mock import EnhancedThermalZoneManager, ThermalState, ThermalZone
from .gpu_metrics import GPUMetrics
from .memory_map import get_memory_map
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor

logger = logging.getLogger(__name__)

class TickEventType(Enum):
    """Types of tick events to track"""
    THERMAL_UPDATE = "thermal_update"
    TRADE_DECISION = "trade_decision"
    GPU_ALLOCATION = "gpu_allocation"
    CPU_ALLOCATION = "cpu_allocation"
    BURST_START = "burst_start"
    BURST_END = "burst_end"
    PROFIT_UPDATE = "profit_update"
    ERROR_EVENT = "error_event"
    SYSTEM_WARNING = "system_warning"

@dataclass
class TickEvent:
    """Individual tick event with detailed information"""
    timestamp: datetime
    event_type: TickEventType
    data: Dict[str, Any]
    thermal_state: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    trade_context: Optional[Dict[str, Any]] = None
    hover_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'data': self.data,
            'thermal_state': self.thermal_state,
            'performance_metrics': self.performance_metrics,
            'trade_context': self.trade_context,
            'hover_info': self.hover_info
        }

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a specific moment"""
    timestamp: datetime
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    thermal_zone: str
    cpu_temp: float
    gpu_temp: float
    drift_coefficient: float
    processing_allocation: Dict[str, float]
    active_trades: int
    profit_pnl: float
    system_health: float

@dataclass
class HoverPortalInfo:
    """Information displayed in hover portals"""
    title: str
    primary_metrics: Dict[str, Any]
    thermal_details: Dict[str, Any]
    performance_details: Dict[str, Any]
    trade_details: Dict[str, Any]
    system_warnings: List[str]
    recommendations: List[str]

class ThermalPerformanceTracker:
    """
    Comprehensive thermal and performance tracking system with visual integration
    """
    
    def __init__(self, 
                 thermal_manager: Optional[EnhancedThermalZoneManager] = None,
                 profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None,
                 gpu_metrics: Optional[GPUMetrics] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize thermal performance tracker
        
        Args:
            thermal_manager: Enhanced thermal zone manager
            profit_coprocessor: Profit trajectory coprocessor
            gpu_metrics: GPU metrics collector
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Core components
        self.thermal_manager = thermal_manager or EnhancedThermalZoneManager()
        self.profit_coprocessor = profit_coprocessor
        self.gpu_metrics = gpu_metrics
        self.memory_map = get_memory_map()
        
        # Tracking data structures
        self.tick_events: deque = deque(maxlen=10000)  # Last 10k events
        self.performance_snapshots: deque = deque(maxlen=1000)  # Last 1k snapshots
        self.thermal_history: deque = deque(maxlen=5000)  # Thermal state history
        self.trade_correlations: Dict[str, List[Dict]] = defaultdict(list)
        
        # Real-time tracking
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        self.active_hover_portals: Dict[str, HoverPortalInfo] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self.cpu_allocation_history: deque = deque(maxlen=1000)
        self.gpu_allocation_history: deque = deque(maxlen=1000)
        self.thermal_efficiency_history: deque = deque(maxlen=1000)
        self.profit_correlation_history: deque = deque(maxlen=1000)
        
        # UI integration
        self.ui_callbacks: Dict[str, Callable] = {}
        self.visualization_data: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_ticks': 0,
            'thermal_events': 0,
            'trade_decisions': 0,
            'burst_events': 0,
            'system_warnings': 0,
            'uptime_start': datetime.now(timezone.utc)
        }
        
        logger.info("ThermalPerformanceTracker initialized")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start real-time monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._capture_performance_snapshot()
                    self._update_visualization_data()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Thermal performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Thermal performance monitoring stopped")
    
    def record_tick_event(self, 
                         event_type: TickEventType,
                         data: Dict[str, Any],
                         trade_context: Optional[Dict[str, Any]] = None) -> None:
        """Record a tick event with full context"""
        with self._lock:
            # Get current thermal state
            thermal_state = None
            if self.thermal_manager:
                thermal_state = self.thermal_manager.get_current_state()
            
            # Get performance metrics
            performance_metrics = self._get_current_performance_metrics()
            
            # Create hover info
            hover_info = self._generate_hover_info(event_type, data, thermal_state, performance_metrics)
            
            # Create tick event
            tick_event = TickEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                data=data,
                thermal_state=thermal_state,
                performance_metrics=performance_metrics,
                trade_context=trade_context,
                hover_info=hover_info
            )
            
            self.tick_events.append(tick_event)
            self.stats['total_ticks'] += 1
            
            # Update specific counters
            if event_type == TickEventType.THERMAL_UPDATE:
                self.stats['thermal_events'] += 1
            elif event_type == TickEventType.TRADE_DECISION:
                self.stats['trade_decisions'] += 1
            elif event_type in [TickEventType.BURST_START, TickEventType.BURST_END]:
                self.stats['burst_events'] += 1
            elif event_type in [TickEventType.ERROR_EVENT, TickEventType.SYSTEM_WARNING]:
                self.stats['system_warnings'] += 1
            
            # Trigger UI callbacks
            self._trigger_ui_callbacks('tick_event', tick_event)
    
    def _capture_performance_snapshot(self) -> None:
        """Capture current performance snapshot"""
        try:
            # Get thermal state
            thermal_state = self.thermal_manager.get_current_state()
            
            # Get system metrics
            cpu_usage = self._get_cpu_usage()
            gpu_usage = self._get_gpu_usage()
            memory_usage = self._get_memory_usage()
            
            # Get profit data
            profit_pnl = 0.0
            if self.profit_coprocessor and self.profit_coprocessor.last_vector:
                profit_pnl = self.profit_coprocessor.smoothed_profit
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                memory_usage=memory_usage,
                thermal_zone=thermal_state.get('thermal_zone', 'unknown'),
                cpu_temp=thermal_state.get('cpu_temperature', 0.0),
                gpu_temp=thermal_state.get('gpu_temperature', 0.0),
                drift_coefficient=thermal_state.get('drift_coefficient', 1.0),
                processing_allocation=thermal_state.get('processing_recommendation', {}),
                active_trades=len(self.trade_correlations),
                profit_pnl=profit_pnl,
                system_health=self._calculate_system_health()
            )
            
            with self._lock:
                self.current_snapshot = snapshot
                self.performance_snapshots.append(snapshot)
                
                # Update allocation histories
                if 'gpu' in snapshot.processing_allocation:
                    self.gpu_allocation_history.append(snapshot.processing_allocation['gpu'])
                if 'cpu' in snapshot.processing_allocation:
                    self.cpu_allocation_history.append(snapshot.processing_allocation['cpu'])
                
                # Update thermal efficiency
                thermal_efficiency = self._calculate_thermal_efficiency(snapshot)
                self.thermal_efficiency_history.append(thermal_efficiency)
                
                # Update profit correlation
                profit_correlation = self._calculate_profit_correlation(snapshot)
                self.profit_correlation_history.append(profit_correlation)
        
        except Exception as e:
            logger.error(f"Error capturing performance snapshot: {e}")
    
    def _generate_hover_info(self, 
                           event_type: TickEventType,
                           data: Dict[str, Any],
                           thermal_state: Optional[Dict[str, Any]],
                           performance_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate hover portal information"""
        hover_info = {
            'event_summary': f"{event_type.value.replace('_', ' ').title()}",
            'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3],
            'details': []
        }
        
        # Add event-specific details
        if event_type == TickEventType.THERMAL_UPDATE:
            if thermal_state:
                hover_info['details'].extend([
                    f"CPU: {thermal_state.get('cpu_temperature', 0):.1f}°C",
                    f"GPU: {thermal_state.get('gpu_temperature', 0):.1f}°C",
                    f"Zone: {thermal_state.get('thermal_zone', 'unknown').upper()}",
                    f"Drift: {thermal_state.get('drift_coefficient', 1.0):.3f}"
                ])
        
        elif event_type == TickEventType.TRADE_DECISION:
            hover_info['details'].extend([
                f"Action: {data.get('action', 'unknown')}",
                f"Amount: ${data.get('amount', 0):,.2f}",
                f"Confidence: {data.get('confidence', 0):.1%}",
                f"Thermal Factor: {data.get('thermal_factor', 1.0):.3f}"
            ])
        
        elif event_type in [TickEventType.GPU_ALLOCATION, TickEventType.CPU_ALLOCATION]:
            hover_info['details'].extend([
                f"Allocation: {data.get('allocation', 0):.1%}",
                f"Load: {data.get('load', 0):.1%}",
                f"Efficiency: {data.get('efficiency', 0):.1%}"
            ])
        
        # Add performance context
        if performance_metrics:
            hover_info['performance'] = {
                'cpu_usage': f"{performance_metrics.get('cpu_usage', 0):.1f}%",
                'gpu_usage': f"{performance_metrics.get('gpu_usage', 0):.1f}%",
                'memory_usage': f"{performance_metrics.get('memory_usage', 0):.1f}%"
            }
        
        # Add thermal context
        if thermal_state:
            hover_info['thermal'] = {
                'zone': thermal_state.get('thermal_zone', 'unknown'),
                'cpu_temp': f"{thermal_state.get('cpu_temperature', 0):.1f}°C",
                'gpu_temp': f"{thermal_state.get('gpu_temperature', 0):.1f}°C"
            }
        
        return hover_info
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for UI visualization"""
        with self._lock:
            # Recent tick events for timeline
            recent_events = list(self.tick_events)[-100:]  # Last 100 events
            
            # Performance snapshots for charts
            recent_snapshots = list(self.performance_snapshots)[-50:]  # Last 50 snapshots
            
            # Allocation data for pie charts
            gpu_allocations = list(self.gpu_allocation_history)[-20:]
            cpu_allocations = list(self.cpu_allocation_history)[-20:]
            
            # Thermal efficiency trend
            thermal_efficiency = list(self.thermal_efficiency_history)[-50:]
            
            return {
                'current_snapshot': asdict(self.current_snapshot) if self.current_snapshot else None,
                'recent_events': [event.to_dict() for event in recent_events],
                'performance_timeline': [
                    {
                        'timestamp': snapshot.timestamp.isoformat(),
                        'cpu_usage': snapshot.cpu_usage,
                        'gpu_usage': snapshot.gpu_usage,
                        'memory_usage': snapshot.memory_usage,
                        'cpu_temp': snapshot.cpu_temp,
                        'gpu_temp': snapshot.gpu_temp,
                        'thermal_zone': snapshot.thermal_zone,
                        'system_health': snapshot.system_health
                    }
                    for snapshot in recent_snapshots
                ],
                'allocation_data': {
                    'gpu_allocations': gpu_allocations,
                    'cpu_allocations': cpu_allocations,
                    'timestamps': [
                        snapshot.timestamp.isoformat() 
                        for snapshot in recent_snapshots[-len(gpu_allocations):]
                    ]
                },
                'thermal_efficiency': thermal_efficiency,
                'statistics': self.stats.copy(),
                'system_health': self._calculate_system_health()
            }
    
    def get_hover_portal_data(self, event_id: str) -> Optional[HoverPortalInfo]:
        """Get detailed hover portal information for a specific event"""
        # Find the event
        for event in reversed(self.tick_events):
            if str(hash(event.timestamp.isoformat() + event.event_type.value)) == event_id:
                return self._create_hover_portal(event)
        return None
    
    def _create_hover_portal(self, event: TickEvent) -> HoverPortalInfo:
        """Create detailed hover portal information"""
        # Primary metrics
        primary_metrics = {
            'Event Type': event.event_type.value.replace('_', ' ').title(),
            'Timestamp': event.timestamp.strftime('%H:%M:%S.%f')[:-3],
            'System Health': f"{self._calculate_system_health():.1%}"
        }
        
        # Thermal details
        thermal_details = {}
        if event.thermal_state:
            thermal_details = {
                'CPU Temperature': f"{event.thermal_state.get('cpu_temperature', 0):.1f}°C",
                'GPU Temperature': f"{event.thermal_state.get('gpu_temperature', 0):.1f}°C",
                'Thermal Zone': event.thermal_state.get('thermal_zone', 'unknown').upper(),
                'Drift Coefficient': f"{event.thermal_state.get('drift_coefficient', 1.0):.3f}",
                'Burst Available': 'Yes' if event.thermal_state.get('burst_available', False) else 'No'
            }
        
        # Performance details
        performance_details = {}
        if event.performance_metrics:
            performance_details = {
                'CPU Usage': f"{event.performance_metrics.get('cpu_usage', 0):.1f}%",
                'GPU Usage': f"{event.performance_metrics.get('gpu_usage', 0):.1f}%",
                'Memory Usage': f"{event.performance_metrics.get('memory_usage', 0):.1f}%"
            }
        
        # Trade details
        trade_details = {}
        if event.trade_context:
            trade_details = {
                'Action': event.trade_context.get('action', 'N/A'),
                'Amount': f"${event.trade_context.get('amount', 0):,.2f}",
                'Confidence': f"{event.trade_context.get('confidence', 0):.1%}"
            }
        
        # System warnings
        warnings = []
        if event.thermal_state:
            if event.thermal_state.get('thermal_zone') in ['hot', 'critical']:
                warnings.append(f"High thermal zone: {event.thermal_state.get('thermal_zone')}")
            if not event.thermal_state.get('burst_available', True):
                warnings.append("Burst processing unavailable")
        
        # Recommendations
        recommendations = self._generate_recommendations(event)
        
        return HoverPortalInfo(
            title=f"{event.event_type.value.replace('_', ' ').title()} Event",
            primary_metrics=primary_metrics,
            thermal_details=thermal_details,
            performance_details=performance_details,
            trade_details=trade_details,
            system_warnings=warnings,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, event: TickEvent) -> List[str]:
        """Generate recommendations based on event data"""
        recommendations = []
        
        if event.thermal_state:
            thermal_zone = event.thermal_state.get('thermal_zone', 'normal')
            
            if thermal_zone == 'hot':
                recommendations.append("Consider reducing GPU allocation")
                recommendations.append("Monitor thermal efficiency closely")
            elif thermal_zone == 'critical':
                recommendations.append("Immediate thermal management required")
                recommendations.append("Switch to CPU-only processing")
            elif thermal_zone == 'cool':
                recommendations.append("Opportunity for increased GPU utilization")
                recommendations.append("Consider burst processing")
        
        if event.performance_metrics:
            cpu_usage = event.performance_metrics.get('cpu_usage', 0)
            gpu_usage = event.performance_metrics.get('gpu_usage', 0)
            
            if cpu_usage > 90:
                recommendations.append("High CPU usage - consider load balancing")
            if gpu_usage > 95:
                recommendations.append("GPU at capacity - thermal monitoring critical")
        
        return recommendations
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback to random simulation
            return 40.0 + np.random.uniform(-10, 20)
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage"""
        if self.gpu_metrics:
            return self.gpu_metrics.get_gpu_utilization()
        else:
            # Fallback to simulation
            return 50.0 + np.random.uniform(-15, 25)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 60.0 + np.random.uniform(-10, 15)
    
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'cpu_usage': self._get_cpu_usage(),
            'gpu_usage': self._get_gpu_usage(),
            'memory_usage': self._get_memory_usage(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        if not self.current_snapshot:
            return 1.0
        
        health_factors = []
        
        # Thermal health (0.3 weight)
        thermal_zones = {'cool': 1.0, 'normal': 0.9, 'warm': 0.7, 'hot': 0.4, 'critical': 0.1}
        thermal_health = thermal_zones.get(self.current_snapshot.thermal_zone, 0.5)
        health_factors.append(thermal_health * 0.3)
        
        # Performance health (0.4 weight)
        cpu_health = max(0.0, 1.0 - (self.current_snapshot.cpu_usage / 100.0))
        gpu_health = max(0.0, 1.0 - (self.current_snapshot.gpu_usage / 100.0))
        memory_health = max(0.0, 1.0 - (self.current_snapshot.memory_usage / 100.0))
        performance_health = (cpu_health + gpu_health + memory_health) / 3.0
        health_factors.append(performance_health * 0.4)
        
        # System stability (0.3 weight)
        uptime = (datetime.now(timezone.utc) - self.stats['uptime_start']).total_seconds()
        stability_health = min(1.0, uptime / 3600.0)  # Full health after 1 hour uptime
        health_factors.append(stability_health * 0.3)
        
        return sum(health_factors)
    
    def _calculate_thermal_efficiency(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate thermal efficiency score"""
        # Lower temperatures and better allocation = higher efficiency
        temp_factor = max(0.0, 1.0 - (max(snapshot.cpu_temp, snapshot.gpu_temp) - 40.0) / 60.0)
        allocation_factor = snapshot.drift_coefficient
        return (temp_factor + allocation_factor) / 2.0
    
    def _calculate_profit_correlation(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate correlation between thermal state and profit"""
        # This would be more sophisticated in practice
        thermal_factor = 1.0 - (max(snapshot.cpu_temp, snapshot.gpu_temp) - 40.0) / 80.0
        profit_factor = max(0.0, min(1.0, snapshot.profit_pnl / 1000.0))  # Normalize to 0-1
        return thermal_factor * profit_factor
    
    def _update_visualization_data(self) -> None:
        """Update cached visualization data"""
        self.visualization_data = self.get_visualization_data()
    
    def register_ui_callback(self, event_type: str, callback: Callable) -> None:
        """Register UI callback for specific events"""
        self.ui_callbacks[event_type] = callback
    
    def _trigger_ui_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger registered UI callbacks"""
        if event_type in self.ui_callbacks:
            try:
                self.ui_callbacks[event_type](data)
            except Exception as e:
                logger.error(f"Error in UI callback for {event_type}: {e}")
    
    def export_data(self, filepath: str, format: str = 'json') -> bool:
        """Export tracking data to file"""
        try:
            data = {
                'metadata': {
                    'export_timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_events': len(self.tick_events),
                    'total_snapshots': len(self.performance_snapshots),
                    'statistics': self.stats
                },
                'tick_events': [event.to_dict() for event in self.tick_events],
                'performance_snapshots': [asdict(snapshot) for snapshot in self.performance_snapshots],
                'visualization_data': self.visualization_data
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

# Factory function for easy integration
def create_thermal_performance_tracker(config: Optional[Dict[str, Any]] = None) -> ThermalPerformanceTracker:
    """Create and configure thermal performance tracker"""
    return ThermalPerformanceTracker(config=config) 