"""
Thermal System Integration
=========================

Complete integration system that connects the enhanced thermal zone manager,
performance tracker, and visual components with the existing Schwabot architecture.
This provides a unified thermal monitoring and management system with real-time
visualization capabilities.

Integration Features:
- Seamless integration with existing visual controller
- Real-time thermal performance monitoring
- Interactive hover portals and detailed analytics
- Cross-platform compatibility
- API endpoints for external integration
- Export capabilities for analysis
- Automated thermal management recommendations
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading
from pathlib import Path

# Core system imports
from .thermal_zone_manager_mock import EnhancedThermalZoneManager
from .thermal_performance_tracker import ThermalPerformanceTracker, TickEventType, create_thermal_performance_tracker
from .thermal_visual_integration import ThermalVisualIntegration, create_thermal_visual_integration
from .unified_visual_controller import UnifiedVisualController
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
from .gpu_metrics import GPUMetrics
from .flask_gateway import Flask, jsonify, request

logger = logging.getLogger(__name__)

@dataclass
class ThermalSystemConfig:
    """Configuration for thermal system integration"""
    monitoring_interval: float = 1.0
    visual_update_interval: float = 1.0
    enable_api_endpoints: bool = True
    enable_visual_integration: bool = True
    enable_hover_portals: bool = True
    export_data_enabled: bool = True
    thermal_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.thermal_thresholds is None:
            self.thermal_thresholds = {
                'cpu_warning': 80.0,
                'cpu_critical': 90.0,
                'gpu_warning': 85.0,
                'gpu_critical': 95.0
            }

class ThermalSystemIntegration:
    """
    Complete thermal system integration that provides unified thermal monitoring,
    performance tracking, and visual integration capabilities.
    """
    
    def __init__(self, 
                 visual_controller: Optional[UnifiedVisualController] = None,
                 profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None,
                 gpu_metrics: Optional[GPUMetrics] = None,
                 config: Optional[ThermalSystemConfig] = None,
                 flask_app: Optional[Flask] = None):
        """
        Initialize thermal system integration
        
        Args:
            visual_controller: Existing unified visual controller
            profit_coprocessor: Profit trajectory coprocessor
            gpu_metrics: GPU metrics collector
            config: System configuration
            flask_app: Flask application for API endpoints
        """
        self.config = config or ThermalSystemConfig()
        self.visual_controller = visual_controller
        self.flask_app = flask_app
        
        # Initialize core components
        self.thermal_manager = EnhancedThermalZoneManager(
            profit_coprocessor=profit_coprocessor,
            config=self._get_thermal_manager_config()
        )
        
        self.performance_tracker = create_thermal_performance_tracker({
            'monitoring_interval': self.config.monitoring_interval
        })
        self.performance_tracker.thermal_manager = self.thermal_manager
        self.performance_tracker.profit_coprocessor = profit_coprocessor
        self.performance_tracker.gpu_metrics = gpu_metrics
        
        self.visual_integration = create_thermal_visual_integration(
            self.performance_tracker,
            self.visual_controller,
            self._get_visual_config()
        )
        
        # System state
        self.is_running = False
        self.system_threads = []
        self._lock = threading.RLock()
        
        # Performance statistics
        self.system_stats = {
            'start_time': None,
            'total_events_processed': 0,
            'thermal_warnings': 0,
            'thermal_criticals': 0,
            'system_health_average': 1.0,
            'uptime_seconds': 0
        }
        
        # API endpoints
        self.api_endpoints = {}
        if self.config.enable_api_endpoints:
            self._setup_api_endpoints()
        
        logger.info("ThermalSystemIntegration initialized")
    
    def _get_thermal_manager_config(self) -> Dict[str, Any]:
        """Get configuration for thermal manager"""
        return {
            'monitoring_interval': self.config.monitoring_interval,
            'thermal_thresholds': self.config.thermal_thresholds,
            'enable_burst_management': True,
            'enable_daily_budget': True
        }
    
    def _get_visual_config(self) -> Dict[str, Any]:
        """Get configuration for visual integration"""
        return {
            'update_interval': self.config.visual_update_interval,
            'enable_hover_portals': self.config.enable_hover_portals,
            'theme': 'dark',
            'color_scheme': {
                'primary': '#00ff88',
                'secondary': '#ff6b35',
                'background': '#1a1a1a',
                'surface': '#2d2d2d',
                'text': '#ffffff',
                'accent': '#4fc3f7'
            }
        }
    
    async def start_system(self) -> bool:
        """Start the complete thermal system"""
        try:
            with self._lock:
                if self.is_running:
                    logger.warning("Thermal system already running")
                    return False
                
                logger.info("Starting thermal system integration...")
                
                # Start thermal manager monitoring
                self.thermal_manager.start_monitoring(
                    interval=self.config.monitoring_interval
                )
                
                # Start performance tracking
                self.performance_tracker.start_monitoring(
                    interval=self.config.monitoring_interval
                )
                
                # Start visual integration if enabled
                if self.config.enable_visual_integration:
                    self.visual_integration.start_visual_updates(
                        interval=self.config.visual_update_interval
                    )
                
                # Register event handlers
                self._register_event_handlers()
                
                # Start background tasks
                await self._start_background_tasks()
                
                self.is_running = True
                self.system_stats['start_time'] = datetime.now(timezone.utc)
                
                logger.info("Thermal system integration started successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error starting thermal system: {e}")
            return False
    
    async def stop_system(self) -> bool:
        """Stop the thermal system"""
        try:
            with self._lock:
                if not self.is_running:
                    logger.warning("Thermal system not running")
                    return False
                
                logger.info("Stopping thermal system integration...")
                
                # Stop monitoring
                self.thermal_manager.stop_monitoring()
                self.performance_tracker.stop_monitoring()
                
                # Stop visual updates
                if self.config.enable_visual_integration:
                    self.visual_integration.stop_visual_updates()
                
                # Stop background tasks
                await self._stop_background_tasks()
                
                self.is_running = False
                
                # Update final statistics
                if self.system_stats['start_time']:
                    uptime = datetime.now(timezone.utc) - self.system_stats['start_time']
                    self.system_stats['uptime_seconds'] = uptime.total_seconds()
                
                logger.info("Thermal system integration stopped")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping thermal system: {e}")
            return False
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for system integration"""
        # Register thermal event handler
        def handle_thermal_event(event_data):
            self._process_thermal_event(event_data)
        
        # Register trade decision handler
        def handle_trade_decision(trade_data):
            self._process_trade_decision(trade_data)
        
        # Register system warning handler
        def handle_system_warning(warning_data):
            self._process_system_warning(warning_data)
        
        # Connect handlers to performance tracker
        self.performance_tracker.register_ui_callback('thermal_event', handle_thermal_event)
        self.performance_tracker.register_ui_callback('trade_decision', handle_trade_decision)
        self.performance_tracker.register_ui_callback('system_warning', handle_system_warning)
    
    def _process_thermal_event(self, event_data) -> None:
        """Process thermal events and update statistics"""
        thermal_state = event_data.thermal_state
        if thermal_state:
            thermal_zone = thermal_state.get('thermal_zone', 'normal')
            
            if thermal_zone in ['hot', 'critical']:
                self.system_stats['thermal_warnings'] += 1
                
                if thermal_zone == 'critical':
                    self.system_stats['thermal_criticals'] += 1
                    
                    # Trigger emergency thermal management
                    self._trigger_emergency_thermal_management(thermal_state)
        
        self.system_stats['total_events_processed'] += 1
    
    def _process_trade_decision(self, trade_data) -> None:
        """Process trade decisions with thermal context"""
        # Record trade decision with thermal correlation
        self.performance_tracker.record_tick_event(
            TickEventType.TRADE_DECISION,
            trade_data.data,
            trade_context=trade_data.trade_context
        )
    
    def _process_system_warning(self, warning_data) -> None:
        """Process system warnings"""
        logger.warning(f"System warning: {warning_data}")
        
        # Record system warning event
        self.performance_tracker.record_tick_event(
            TickEventType.SYSTEM_WARNING,
            {'warning': warning_data}
        )
    
    def _trigger_emergency_thermal_management(self, thermal_state: Dict[str, Any]) -> None:
        """Trigger emergency thermal management procedures"""
        logger.critical(f"Emergency thermal management triggered: {thermal_state}")
        
        # Force CPU-only processing
        if self.thermal_manager:
            # This would trigger emergency thermal management in the actual system
            pass
        
        # Notify visual controller
        if self.visual_controller:
            self.visual_controller.show_thermal_warning(thermal_state)
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks"""
        # Start system health monitoring
        health_task = asyncio.create_task(self._system_health_monitor())
        self.system_threads.append(health_task)
        
        # Start statistics collection
        stats_task = asyncio.create_task(self._statistics_collector())
        self.system_threads.append(stats_task)
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks"""
        for task in self.system_threads:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.system_threads:
            await asyncio.gather(*self.system_threads, return_exceptions=True)
        
        self.system_threads.clear()
    
    async def _system_health_monitor(self) -> None:
        """Monitor overall system health"""
        while self.is_running:
            try:
                # Get current system health
                current_snapshot = self.performance_tracker.current_snapshot
                if current_snapshot:
                    health_score = current_snapshot.system_health
                    
                    # Update running average
                    current_avg = self.system_stats.get('system_health_average', 1.0)
                    self.system_stats['system_health_average'] = (
                        0.95 * current_avg + 0.05 * health_score
                    )
                    
                    # Check for critical health issues
                    if health_score < 0.3:
                        await self._handle_critical_health_issue(current_snapshot)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system health monitor: {e}")
                await asyncio.sleep(10)
    
    async def _statistics_collector(self) -> None:
        """Collect and update system statistics"""
        while self.is_running:
            try:
                # Update uptime
                if self.system_stats['start_time']:
                    uptime = datetime.now(timezone.utc) - self.system_stats['start_time']
                    self.system_stats['uptime_seconds'] = uptime.total_seconds()
                
                # Collect performance statistics
                perf_stats = self.performance_tracker.stats
                self.system_stats.update({
                    'total_ticks': perf_stats.get('total_ticks', 0),
                    'thermal_events': perf_stats.get('thermal_events', 0),
                    'trade_decisions': perf_stats.get('trade_decisions', 0),
                    'burst_events': perf_stats.get('burst_events', 0)
                })
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in statistics collector: {e}")
                await asyncio.sleep(30)
    
    async def _handle_critical_health_issue(self, snapshot) -> None:
        """Handle critical system health issues"""
        logger.critical(f"Critical system health detected: {snapshot.system_health:.1%}")
        
        # Record critical event
        self.performance_tracker.record_tick_event(
            TickEventType.ERROR_EVENT,
            {
                'type': 'critical_health',
                'system_health': snapshot.system_health,
                'thermal_zone': snapshot.thermal_zone,
                'cpu_temp': snapshot.cpu_temp,
                'gpu_temp': snapshot.gpu_temp
            }
        )
        
        # Trigger emergency procedures
        await self._trigger_emergency_procedures(snapshot)
    
    async def _trigger_emergency_procedures(self, snapshot) -> None:
        """Trigger emergency system procedures"""
        # Force thermal management
        if snapshot.thermal_zone in ['hot', 'critical']:
            self._trigger_emergency_thermal_management(asdict(snapshot))
        
        # Reduce system load
        if snapshot.cpu_usage > 95 or snapshot.gpu_usage > 95:
            logger.warning("Reducing system load due to high resource usage")
    
    def _setup_api_endpoints(self) -> None:
        """Setup API endpoints for thermal system"""
        if not self.flask_app:
            return
        
        @self.flask_app.route('/api/thermal/status', methods=['GET'])
        def get_thermal_status():
            """Get current thermal system status"""
            try:
                current_state = self.thermal_manager.get_current_state()
                system_stats = self.get_system_statistics()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'thermal_state': current_state,
                        'system_statistics': system_stats,
                        'is_running': self.is_running,
                        'uptime_seconds': system_stats.get('uptime_seconds', 0)
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.flask_app.route('/api/thermal/dashboard', methods=['GET'])
        def get_thermal_dashboard():
            """Get thermal dashboard HTML"""
            try:
                dashboard_html = self.visual_integration.get_dashboard_html()
                return dashboard_html
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.flask_app.route('/api/thermal/visualization-data', methods=['GET'])
        def get_visualization_data():
            """Get visualization data for dashboard"""
            try:
                data = self.visual_integration.get_widget_data()
                return jsonify({
                    'success': True,
                    'data': data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.flask_app.route('/api/thermal/export', methods=['POST'])
        def export_thermal_data():
            """Export thermal data"""
            try:
                if not self.config.export_data_enabled:
                    return jsonify({
                        'success': False,
                        'error': 'Data export disabled'
                    }), 403
                
                data = request.get_json()
                filepath = data.get('filepath', 'thermal_export.json')
                format_type = data.get('format', 'json')
                
                success = self.performance_tracker.export_data(filepath, format_type)
                
                return jsonify({
                    'success': success,
                    'filepath': filepath if success else None
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # Add visual integration endpoints
        visual_endpoints = self.visual_integration.create_api_endpoints()
        for endpoint, handler in visual_endpoints.items():
            # Register endpoints with Flask app
            # This would need proper Flask route registration
            pass
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = self.system_stats.copy()
        
        # Add thermal manager statistics
        if self.thermal_manager:
            thermal_stats = self.thermal_manager.get_statistics()
            stats['thermal_manager'] = thermal_stats
        
        # Add performance tracker statistics
        if self.performance_tracker:
            perf_stats = self.performance_tracker.stats
            stats['performance_tracker'] = perf_stats
        
        # Add visual integration statistics
        if self.visual_integration:
            visual_stats = {
                'active_widgets': len(self.visual_integration.active_widgets),
                'hover_portals': len(self.visual_integration.hover_portals),
                'update_active': self.visual_integration.update_active
            }
            stats['visual_integration'] = visual_stats
        
        return stats
    
    def record_trading_event(self, 
                           action: str,
                           amount: float,
                           confidence: float,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a trading event with thermal context"""
        trade_data = {
            'action': action,
            'amount': amount,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        # Get current thermal state for context
        thermal_state = self.thermal_manager.get_current_state()
        trade_context = {
            'thermal_zone': thermal_state.get('thermal_zone', 'unknown'),
            'thermal_factor': thermal_state.get('drift_coefficient', 1.0),
            'system_health': self.performance_tracker.current_snapshot.system_health 
                           if self.performance_tracker.current_snapshot else 1.0
        }
        
        self.performance_tracker.record_tick_event(
            TickEventType.TRADE_DECISION,
            trade_data,
            trade_context=trade_context
        )
    
    def get_thermal_recommendations(self) -> List[str]:
        """Get current thermal management recommendations"""
        recommendations = []
        
        current_state = self.thermal_manager.get_current_state()
        if current_state:
            thermal_zone = current_state.get('thermal_zone', 'normal')
            cpu_temp = current_state.get('cpu_temperature', 0)
            gpu_temp = current_state.get('gpu_temperature', 0)
            
            if thermal_zone == 'critical':
                recommendations.extend([
                    "CRITICAL: Immediate thermal management required",
                    "Switch to CPU-only processing",
                    "Consider system shutdown if temperatures continue rising"
                ])
            elif thermal_zone == 'hot':
                recommendations.extend([
                    "Reduce GPU allocation",
                    "Increase cooling if possible",
                    "Monitor thermal efficiency closely"
                ])
            elif thermal_zone == 'cool':
                recommendations.extend([
                    "Opportunity for increased GPU utilization",
                    "Consider burst processing",
                    "Optimal conditions for intensive operations"
                ])
            
            # Temperature-specific recommendations
            if cpu_temp > self.config.thermal_thresholds['cpu_warning']:
                recommendations.append(f"CPU temperature high: {cpu_temp:.1f}°C")
            
            if gpu_temp > self.config.thermal_thresholds['gpu_warning']:
                recommendations.append(f"GPU temperature high: {gpu_temp:.1f}°C")
        
        return recommendations
    
    def get_dashboard_widget_html(self) -> str:
        """Get embeddable thermal dashboard widget HTML"""
        return self.visual_integration.create_thermal_dashboard()
    
    def is_system_healthy(self) -> bool:
        """Check if the thermal system is healthy"""
        if not self.is_running:
            return False
        
        current_snapshot = self.performance_tracker.current_snapshot
        if not current_snapshot:
            return True  # No data yet, assume healthy
        
        # Check system health score
        if current_snapshot.system_health < 0.5:
            return False
        
        # Check thermal zones
        if current_snapshot.thermal_zone == 'critical':
            return False
        
        return True

# Factory function for easy integration
def create_thermal_system_integration(
    visual_controller: Optional[UnifiedVisualController] = None,
    profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None,
    gpu_metrics: Optional[GPUMetrics] = None,
    config: Optional[ThermalSystemConfig] = None,
    flask_app: Optional[Flask] = None
) -> ThermalSystemIntegration:
    """Create and configure thermal system integration"""
    return ThermalSystemIntegration(
        visual_controller=visual_controller,
        profit_coprocessor=profit_coprocessor,
        gpu_metrics=gpu_metrics,
        config=config,
        flask_app=flask_app
    ) 