"""
Thermal Visual Integration
=========================

Visual integration component for thermal performance tracking that provides
real-time widgets, hover portals, and interactive visualizations for the
Schwabot UI system. This component creates modern, cross-platform compatible
visual elements that integrate seamlessly with the existing visual controller.

Features:
- Real-time thermal monitoring widgets
- Interactive hover portals with detailed information
- CPU/GPU allocation pie charts and timeline graphs
- Thermal efficiency indicators
- Trade correlation visualizations
- System health dashboard
- Cross-platform compatibility (Windows, Mac, Linux)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading
import asyncio
from pathlib import Path

# Core system imports
from .thermal_performance_tracker import ThermalPerformanceTracker, TickEventType, HoverPortalInfo
from .unified_visual_controller import UnifiedVisualController

logger = logging.getLogger(__name__)

@dataclass
class ThermalWidget:
    """Configuration for thermal monitoring widgets"""
    widget_id: str
    widget_type: str  # 'gauge', 'chart', 'timeline', 'portal'
    title: str
    position: Tuple[int, int]  # (x, y)
    size: Tuple[int, int]  # (width, height)
    update_interval: float = 1.0
    config: Dict[str, Any] = None

@dataclass
class HoverPortal:
    """Interactive hover portal configuration"""
    portal_id: str
    trigger_element: str
    content_template: str
    position_mode: str = 'mouse'  # 'mouse', 'fixed', 'relative'
    animation: str = 'fade'  # 'fade', 'slide', 'scale'
    delay_ms: int = 300

class ThermalVisualIntegration:
    """
    Visual integration system for thermal performance tracking
    """
    
    def __init__(self, 
                 performance_tracker: ThermalPerformanceTracker,
                 visual_controller: Optional[UnifiedVisualController] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize thermal visual integration
        
        Args:
            performance_tracker: Thermal performance tracker instance
            visual_controller: Unified visual controller for UI integration
            config: Visual configuration dictionary
        """
        self.performance_tracker = performance_tracker
        self.visual_controller = visual_controller
        self.config = config or self._default_config()
        
        # Widget management
        self.active_widgets: Dict[str, ThermalWidget] = {}
        self.hover_portals: Dict[str, HoverPortal] = {}
        self.widget_data_cache: Dict[str, Any] = {}
        
        # Visual elements
        self.thermal_dashboard_html = ""
        self.widget_templates = self._load_widget_templates()
        self.css_styles = self._load_css_styles()
        self.javascript_handlers = self._load_javascript_handlers()
        
        # Real-time updates
        self.update_active = False
        self.update_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Event handling
        self.ui_event_handlers: Dict[str, Callable] = {}
        
        # Register with performance tracker
        self.performance_tracker.register_ui_callback('tick_event', self._handle_tick_event)
        
        logger.info("ThermalVisualIntegration initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default visual configuration"""
        return {
            'theme': 'dark',
            'color_scheme': {
                'primary': '#00ff88',
                'secondary': '#ff6b35',
                'background': '#1a1a1a',
                'surface': '#2d2d2d',
                'text': '#ffffff',
                'accent': '#4fc3f7'
            },
            'animation_duration': 300,
            'update_interval': 1.0,
            'hover_delay': 300,
            'chart_options': {
                'responsive': True,
                'maintain_aspect_ratio': False,
                'animation': {'duration': 500}
            }
        }
    
    def create_thermal_dashboard(self) -> str:
        """Create comprehensive thermal dashboard HTML"""
        dashboard_html = f"""
        <div id="thermal-dashboard" class="thermal-dashboard">
            <div class="dashboard-header">
                <h2>Thermal Performance Monitor</h2>
                <div class="system-status" id="system-status">
                    <span class="status-indicator" id="status-indicator"></span>
                    <span class="status-text" id="status-text">Initializing...</span>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <!-- Real-time metrics -->
                <div class="widget-container" id="realtime-metrics">
                    <div class="widget-header">
                        <h3>Real-time Metrics</h3>
                        <div class="widget-controls">
                            <button class="btn-toggle" data-widget="realtime-metrics">📊</button>
                        </div>
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-card" data-hover-portal="cpu-temp">
                            <div class="metric-value" id="cpu-temp-value">--°C</div>
                            <div class="metric-label">CPU Temperature</div>
                            <div class="metric-trend" id="cpu-temp-trend"></div>
                        </div>
                        <div class="metric-card" data-hover-portal="gpu-temp">
                            <div class="metric-value" id="gpu-temp-value">--°C</div>
                            <div class="metric-label">GPU Temperature</div>
                            <div class="metric-trend" id="gpu-temp-trend"></div>
                        </div>
                        <div class="metric-card" data-hover-portal="thermal-zone">
                            <div class="metric-value" id="thermal-zone-value">--</div>
                            <div class="metric-label">Thermal Zone</div>
                            <div class="zone-indicator" id="zone-indicator"></div>
                        </div>
                        <div class="metric-card" data-hover-portal="system-health">
                            <div class="metric-value" id="system-health-value">--%</div>
                            <div class="metric-label">System Health</div>
                            <div class="health-bar" id="health-bar"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Performance allocation -->
                <div class="widget-container" id="allocation-widget">
                    <div class="widget-header">
                        <h3>Processing Allocation</h3>
                    </div>
                    <div class="allocation-display">
                        <canvas id="allocation-chart" width="300" height="200"></canvas>
                        <div class="allocation-details">
                            <div class="allocation-item">
                                <span class="allocation-color gpu-color"></span>
                                <span class="allocation-label">GPU: <span id="gpu-allocation">--%</span></span>
                            </div>
                            <div class="allocation-item">
                                <span class="allocation-color cpu-color"></span>
                                <span class="allocation-label">CPU: <span id="cpu-allocation">--%</span></span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Thermal timeline -->
                <div class="widget-container full-width" id="thermal-timeline">
                    <div class="widget-header">
                        <h3>Thermal Timeline</h3>
                        <div class="timeline-controls">
                            <select id="timeline-range">
                                <option value="1m">1 Minute</option>
                                <option value="5m" selected>5 Minutes</option>
                                <option value="15m">15 Minutes</option>
                                <option value="1h">1 Hour</option>
                            </select>
                        </div>
                    </div>
                    <div class="timeline-container">
                        <canvas id="thermal-timeline-chart" width="800" height="300"></canvas>
                    </div>
                </div>
                
                <!-- Event timeline -->
                <div class="widget-container full-width" id="event-timeline">
                    <div class="widget-header">
                        <h3>System Events</h3>
                        <div class="event-filters">
                            <button class="filter-btn active" data-filter="all">All</button>
                            <button class="filter-btn" data-filter="thermal">Thermal</button>
                            <button class="filter-btn" data-filter="trade">Trades</button>
                            <button class="filter-btn" data-filter="burst">Bursts</button>
                        </div>
                    </div>
                    <div class="event-timeline-container">
                        <div class="event-list" id="event-list">
                            <!-- Events will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Hover portals -->
        <div id="hover-portal" class="hover-portal" style="display: none;">
            <div class="portal-header">
                <h4 id="portal-title">Information</h4>
                <button class="portal-close" onclick="hideHoverPortal()">×</button>
            </div>
            <div class="portal-content" id="portal-content">
                <!-- Content will be populated dynamically -->
            </div>
        </div>
        """
        
        return dashboard_html
    
    def _load_css_styles(self) -> str:
        """Load CSS styles for thermal dashboard"""
        return """
        <style>
        .thermal-dashboard {
            background: var(--bg-color, #1a1a1a);
            color: var(--text-color, #ffffff);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--accent-color, #4fc3f7);
        }
        
        .dashboard-header h2 {
            margin: 0;
            color: var(--primary-color, #00ff88);
            font-weight: 600;
        }
        
        .system-status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--primary-color, #00ff88);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            grid-auto-rows: min-content;
        }
        
        .widget-container {
            background: var(--surface-color, #2d2d2d);
            border-radius: 8px;
            padding: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .widget-container.full-width {
            grid-column: 1 / -1;
        }
        
        .widget-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .widget-header h3 {
            margin: 0;
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .metric-card:hover {
            border-color: var(--accent-color, #4fc3f7);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 195, 247, 0.2);
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: 600;
            color: var(--primary-color, #00ff88);
            margin-bottom: 4px;
        }
        
        .metric-label {
            font-size: 0.85em;
            opacity: 0.8;
            margin-bottom: 8px;
        }
        
        .allocation-display {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .allocation-details {
            flex: 1;
        }
        
        .allocation-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .allocation-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        
        .allocation-color.gpu-color {
            background: var(--secondary-color, #ff6b35);
        }
        
        .allocation-color.cpu-color {
            background: var(--accent-color, #4fc3f7);
        }
        
        .timeline-container, .event-timeline-container {
            position: relative;
            overflow: hidden;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.2);
        }
        
        .event-list {
            max-height: 200px;
            overflow-y: auto;
            padding: 8px;
        }
        
        .event-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 12px;
            margin-bottom: 4px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        
        .event-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .event-type {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .event-type.thermal { background: var(--secondary-color, #ff6b35); }
        .event-type.trade { background: var(--primary-color, #00ff88); }
        .event-type.burst { background: var(--accent-color, #4fc3f7); }
        
        .hover-portal {
            position: fixed;
            background: var(--surface-color, #2d2d2d);
            border: 1px solid var(--accent-color, #4fc3f7);
            border-radius: 8px;
            padding: 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            z-index: 1000;
            max-width: 320px;
            animation: portalFadeIn 0.3s ease;
        }
        
        @keyframes portalFadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        .portal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            background: var(--accent-color, #4fc3f7);
            border-radius: 7px 7px 0 0;
        }
        
        .portal-header h4 {
            margin: 0;
            color: white;
            font-size: 0.9em;
        }
        
        .portal-close {
            background: none;
            border: none;
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .portal-content {
            padding: 16px;
        }
        
        .portal-section {
            margin-bottom: 16px;
        }
        
        .portal-section:last-child {
            margin-bottom: 0;
        }
        
        .portal-section h5 {
            margin: 0 0 8px 0;
            font-size: 0.85em;
            color: var(--accent-color, #4fc3f7);
            text-transform: uppercase;
            font-weight: 600;
        }
        
        .portal-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.85em;
        }
        
        .portal-metric-label {
            opacity: 0.8;
        }
        
        .portal-metric-value {
            font-weight: 500;
            color: var(--primary-color, #00ff88);
        }
        
        .filter-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid transparent;
            color: var(--text-color, #ffffff);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            margin-right: 8px;
            transition: all 0.2s ease;
        }
        
        .filter-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .filter-btn.active {
            background: var(--accent-color, #4fc3f7);
            border-color: var(--accent-color, #4fc3f7);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .allocation-display {
                flex-direction: column;
                align-items: center;
            }
        }
        </style>
        """
    
    def _load_javascript_handlers(self) -> str:
        """Load JavaScript handlers for thermal dashboard"""
        return """
        <script>
        class ThermalDashboard {
            constructor() {
                this.charts = {};
                this.eventFilters = new Set(['all']);
                this.updateInterval = null;
                this.hoverPortal = null;
                this.init();
            }
            
            init() {
                this.setupEventListeners();
                this.initializeCharts();
                this.startUpdates();
            }
            
            setupEventListeners() {
                // Hover portal events
                document.querySelectorAll('[data-hover-portal]').forEach(element => {
                    element.addEventListener('mouseenter', (e) => {
                        const portalType = e.target.getAttribute('data-hover-portal');
                        this.showHoverPortal(e.target, portalType);
                    });
                    
                    element.addEventListener('mouseleave', () => {
                        this.hideHoverPortal();
                    });
                });
                
                // Filter buttons
                document.querySelectorAll('.filter-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        this.setEventFilter(e.target.getAttribute('data-filter'));
                    });
                });
                
                // Timeline range selector
                const rangeSelector = document.getElementById('timeline-range');
                if (rangeSelector) {
                    rangeSelector.addEventListener('change', (e) => {
                        this.updateTimelineRange(e.target.value);
                    });
                }
            }
            
            initializeCharts() {
                // Initialize allocation chart
                const allocationCanvas = document.getElementById('allocation-chart');
                if (allocationCanvas) {
                    this.charts.allocation = new Chart(allocationCanvas.getContext('2d'), {
                        type: 'doughnut',
                        data: {
                            labels: ['GPU', 'CPU'],
                            datasets: [{
                                data: [50, 50],
                                backgroundColor: ['#ff6b35', '#4fc3f7'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false }
                            }
                        }
                    });
                }
                
                // Initialize thermal timeline chart
                const timelineCanvas = document.getElementById('thermal-timeline-chart');
                if (timelineCanvas) {
                    this.charts.timeline = new Chart(timelineCanvas.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'CPU Temperature',
                                data: [],
                                borderColor: '#4fc3f7',
                                backgroundColor: 'rgba(79, 195, 247, 0.1)',
                                fill: false
                            }, {
                                label: 'GPU Temperature',
                                data: [],
                                borderColor: '#ff6b35',
                                backgroundColor: 'rgba(255, 107, 53, 0.1)',
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    title: { display: true, text: 'Temperature (°C)' }
                                }
                            }
                        }
                    });
                }
            }
            
            async updateDashboard() {
                try {
                    const response = await fetch('/api/thermal/visualization-data');
                    const data = await response.json();
                    
                    if (data.success) {
                        this.updateMetrics(data.data);
                        this.updateCharts(data.data);
                        this.updateEventList(data.data.recent_events || []);
                    }
                } catch (error) {
                    console.error('Error updating thermal dashboard:', error);
                }
            }
            
            updateMetrics(data) {
                const snapshot = data.current_snapshot;
                if (!snapshot) return;
                
                // Update temperature displays
                document.getElementById('cpu-temp-value').textContent = 
                    snapshot.cpu_temp?.toFixed(1) + '°C' || '--°C';
                document.getElementById('gpu-temp-value').textContent = 
                    snapshot.gpu_temp?.toFixed(1) + '°C' || '--°C';
                
                // Update thermal zone
                document.getElementById('thermal-zone-value').textContent = 
                    snapshot.thermal_zone?.toUpperCase() || '--';
                
                // Update system health
                const healthPercent = (snapshot.system_health * 100).toFixed(0);
                document.getElementById('system-health-value').textContent = healthPercent + '%';
                
                // Update status indicator
                this.updateStatusIndicator(snapshot.thermal_zone, snapshot.system_health);
            }
            
            updateStatusIndicator(thermalZone, systemHealth) {
                const indicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                
                if (systemHealth > 0.8) {
                    indicator.style.background = '#00ff88';
                    statusText.textContent = 'Optimal';
                } else if (systemHealth > 0.6) {
                    indicator.style.background = '#ffeb3b';
                    statusText.textContent = 'Good';
                } else if (systemHealth > 0.4) {
                    indicator.style.background = '#ff9800';
                    statusText.textContent = 'Warning';
                } else {
                    indicator.style.background = '#f44336';
                    statusText.textContent = 'Critical';
                }
            }
            
            updateCharts(data) {
                // Update allocation chart
                if (this.charts.allocation && data.allocation_data) {
                    const gpuAlloc = data.allocation_data.gpu_allocations;
                    const cpuAlloc = data.allocation_data.cpu_allocations;
                    
                    if (gpuAlloc.length > 0 && cpuAlloc.length > 0) {
                        const latestGpu = gpuAlloc[gpuAlloc.length - 1] * 100;
                        const latestCpu = cpuAlloc[cpuAlloc.length - 1] * 100;
                        
                        this.charts.allocation.data.datasets[0].data = [latestGpu, latestCpu];
                        this.charts.allocation.update();
                        
                        // Update allocation text
                        document.getElementById('gpu-allocation').textContent = latestGpu.toFixed(0) + '%';
                        document.getElementById('cpu-allocation').textContent = latestCpu.toFixed(0) + '%';
                    }
                }
                
                // Update timeline chart
                if (this.charts.timeline && data.performance_timeline) {
                    const timeline = data.performance_timeline.slice(-50); // Last 50 points
                    const labels = timeline.map(point => 
                        new Date(point.timestamp).toLocaleTimeString()
                    );
                    const cpuTemps = timeline.map(point => point.cpu_temp);
                    const gpuTemps = timeline.map(point => point.gpu_temp);
                    
                    this.charts.timeline.data.labels = labels;
                    this.charts.timeline.data.datasets[0].data = cpuTemps;
                    this.charts.timeline.data.datasets[1].data = gpuTemps;
                    this.charts.timeline.update();
                }
            }
            
            updateEventList(events) {
                const eventList = document.getElementById('event-list');
                if (!eventList) return;
                
                // Filter events based on active filters
                const filteredEvents = events.filter(event => {
                    if (this.eventFilters.has('all')) return true;
                    return this.eventFilters.has(event.event_type.split('_')[0]);
                });
                
                // Clear and populate event list
                eventList.innerHTML = '';
                filteredEvents.slice(-20).forEach(event => {
                    const eventElement = this.createEventElement(event);
                    eventList.appendChild(eventElement);
                });
            }
            
            createEventElement(event) {
                const div = document.createElement('div');
                div.className = 'event-item';
                div.setAttribute('data-event-id', event.timestamp);
                
                const eventType = event.event_type.split('_')[0];
                const timestamp = new Date(event.timestamp).toLocaleTimeString();
                
                div.innerHTML = `
                    <span class="event-type ${eventType}">${eventType}</span>
                    <span class="event-time">${timestamp}</span>
                    <span class="event-description">${event.hover_info?.event_summary || event.event_type}</span>
                `;
                
                div.addEventListener('click', () => {
                    this.showEventDetails(event);
                });
                
                return div;
            }
            
            showHoverPortal(element, portalType) {
                const portal = document.getElementById('hover-portal');
                if (!portal) return;
                
                // Get portal content based on type
                this.getPortalContent(portalType).then(content => {
                    document.getElementById('portal-title').textContent = content.title;
                    document.getElementById('portal-content').innerHTML = content.html;
                    
                    // Position portal
                    const rect = element.getBoundingClientRect();
                    portal.style.left = (rect.right + 10) + 'px';
                    portal.style.top = rect.top + 'px';
                    portal.style.display = 'block';
                    
                    this.hoverPortal = portal;
                });
            }
            
            hideHoverPortal() {
                if (this.hoverPortal) {
                    this.hoverPortal.style.display = 'none';
                    this.hoverPortal = null;
                }
            }
            
            async getPortalContent(portalType) {
                // This would fetch detailed information for the hover portal
                // For now, return mock content
                return {
                    title: portalType.replace('-', ' ').toUpperCase(),
                    html: `
                        <div class="portal-section">
                            <h5>Current Status</h5>
                            <div class="portal-metric">
                                <span class="portal-metric-label">Status:</span>
                                <span class="portal-metric-value">Active</span>
                            </div>
                        </div>
                    `
                };
            }
            
            setEventFilter(filter) {
                // Update filter buttons
                document.querySelectorAll('.filter-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
                
                // Update filter set
                this.eventFilters.clear();
                this.eventFilters.add(filter);
                
                // Refresh event list
                this.updateDashboard();
            }
            
            startUpdates() {
                this.updateInterval = setInterval(() => {
                    this.updateDashboard();
                }, 1000);
                
                // Initial update
                this.updateDashboard();
            }
            
            stopUpdates() {
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                    this.updateInterval = null;
                }
            }
        }
        
        // Global functions
        function hideHoverPortal() {
            const portal = document.getElementById('hover-portal');
            if (portal) {
                portal.style.display = 'none';
            }
        }
        
        // Initialize dashboard when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            window.thermalDashboard = new ThermalDashboard();
        });
        </script>
        """
    
    def _load_widget_templates(self) -> Dict[str, str]:
        """Load widget templates"""
        return {
            'gauge': """
                <div class="gauge-widget" id="{widget_id}">
                    <div class="gauge-container">
                        <canvas id="{widget_id}-canvas" width="200" height="200"></canvas>
                        <div class="gauge-value">{value}</div>
                        <div class="gauge-label">{label}</div>
                    </div>
                </div>
            """,
            'timeline': """
                <div class="timeline-widget" id="{widget_id}">
                    <canvas id="{widget_id}-canvas" width="400" height="200"></canvas>
                </div>
            """,
            'portal': """
                <div class="portal-widget" id="{widget_id}" data-hover-portal="{portal_type}">
                    <div class="portal-trigger">
                        <i class="icon-{icon}"></i>
                        <span>{label}</span>
                    </div>
                </div>
            """
        }
    
    def _handle_tick_event(self, tick_event) -> None:
        """Handle tick events from performance tracker"""
        # Update widget data cache
        with self._lock:
            event_data = {
                'timestamp': tick_event.timestamp.isoformat(),
                'type': tick_event.event_type.value,
                'data': tick_event.data,
                'hover_info': tick_event.hover_info
            }
            
            # Store in cache for UI updates
            if 'recent_events' not in self.widget_data_cache:
                self.widget_data_cache['recent_events'] = []
            
            self.widget_data_cache['recent_events'].append(event_data)
            
            # Keep only last 100 events
            if len(self.widget_data_cache['recent_events']) > 100:
                self.widget_data_cache['recent_events'] = self.widget_data_cache['recent_events'][-100:]
    
    def get_dashboard_html(self) -> str:
        """Get complete dashboard HTML with styles and scripts"""
        dashboard_html = self.create_thermal_dashboard()
        css_styles = self._load_css_styles()
        js_handlers = self._load_javascript_handlers()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Thermal Performance Monitor</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {css_styles}
        </head>
        <body>
            {dashboard_html}
            {js_handlers}
        </body>
        </html>
        """
    
    def get_widget_data(self) -> Dict[str, Any]:
        """Get current widget data for UI updates"""
        visualization_data = self.performance_tracker.get_visualization_data()
        
        with self._lock:
            # Merge with cached data
            widget_data = {
                **visualization_data,
                'cached_events': self.widget_data_cache.get('recent_events', [])
            }
        
        return widget_data
    
    def create_api_endpoints(self) -> Dict[str, Callable]:
        """Create API endpoints for thermal dashboard"""
        def get_visualization_data():
            return {
                'success': True,
                'data': self.get_widget_data()
            }
        
        def get_hover_portal_data(event_id: str):
            portal_info = self.performance_tracker.get_hover_portal_data(event_id)
            if portal_info:
                return {
                    'success': True,
                    'data': asdict(portal_info)
                }
            return {'success': False, 'error': 'Event not found'}
        
        return {
            '/api/thermal/visualization-data': get_visualization_data,
            '/api/thermal/hover-portal/<event_id>': get_hover_portal_data
        }
    
    def start_visual_updates(self, interval: float = 1.0) -> None:
        """Start visual update loop"""
        if self.update_active:
            return
        
        self.update_active = True
        
        def update_loop():
            while self.update_active:
                try:
                    # Update widget data cache
                    self.widget_data_cache.update(self.get_widget_data())
                    
                    # Trigger visual controller updates if available
                    if self.visual_controller:
                        self.visual_controller.update_thermal_widgets(self.widget_data_cache)
                    
                    threading.Event().wait(interval)
                except Exception as e:
                    logger.error(f"Error in visual update loop: {e}")
                    threading.Event().wait(interval)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        logger.info(f"Visual updates started (interval: {interval}s)")
    
    def stop_visual_updates(self) -> None:
        """Stop visual update loop"""
        self.update_active = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        logger.info("Visual updates stopped")

# Factory function for easy integration
def create_thermal_visual_integration(
    performance_tracker: ThermalPerformanceTracker,
    visual_controller: Optional[UnifiedVisualController] = None,
    config: Optional[Dict[str, Any]] = None
) -> ThermalVisualIntegration:
    """Create and configure thermal visual integration"""
    return ThermalVisualIntegration(performance_tracker, visual_controller, config) 