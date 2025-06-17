"""
BTC Processor Web UI
====================

Web-based interface for controlling BTC processor features, monitoring system resources,
and managing load balancing to prevent system overload during live testing.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import asyncio
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Any
import json

from core.btc_processor_controller import BTCProcessorController
from core.btc_data_processor import BTCDataProcessor

logger = logging.getLogger(__name__)

class BTCProcessorUI:
    """Web-based UI for BTC processor control"""
    
    def __init__(self, processor=None, host='localhost', port=5000):
        self.app = Flask(__name__, template_folder='templates')
        self.host = host
        self.port = port
        self.processor = processor
        self.controller = BTCProcessorController(processor)
        self.is_running = False
        self.background_thread = None
        
        # Setup Flask routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes for the web interface"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('btc_processor_dashboard.html')
            
        @self.app.route('/api/status')
        def get_status():
            """Get current processor and system status"""
            try:
                status = self.controller.get_current_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/features/<feature_name>/enable', methods=['POST'])
        def enable_feature(feature_name):
            """Enable a specific feature"""
            try:
                asyncio.run(self.controller.enable_feature(feature_name))
                return jsonify({'success': True, 'message': f'Enabled {feature_name}'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/features/<feature_name>/disable', methods=['POST'])
        def disable_feature(feature_name):
            """Disable a specific feature"""
            try:
                asyncio.run(self.controller.disable_feature(feature_name))
                return jsonify({'success': True, 'message': f'Disabled {feature_name}'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/features/analysis/disable-all', methods=['POST'])
        def disable_all_analysis():
            """Disable all analysis features"""
            try:
                asyncio.run(self.controller.disable_all_analysis_features())
                return jsonify({'success': True, 'message': 'Disabled all analysis features'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/features/analysis/enable-all', methods=['POST'])
        def enable_all_analysis():
            """Enable all analysis features"""
            try:
                asyncio.run(self.controller.enable_all_analysis_features())
                return jsonify({'success': True, 'message': 'Enabled all analysis features'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/emergency-cleanup', methods=['POST'])
        def emergency_cleanup():
            """Trigger emergency cleanup"""
            try:
                asyncio.run(self.controller._emergency_memory_cleanup())
                return jsonify({'success': True, 'message': 'Emergency cleanup completed'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/thresholds', methods=['GET', 'POST'])
        def manage_thresholds():
            """Get or update system thresholds"""
            if request.method == 'GET':
                return jsonify(self.controller.system_thresholds)
            else:
                try:
                    new_thresholds = request.json
                    self.controller.update_thresholds(new_thresholds)
                    return jsonify({'success': True, 'message': 'Thresholds updated'})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
                    
        @self.app.route('/api/configuration', methods=['GET', 'POST'])
        def manage_configuration():
            """Get or update processor configuration"""
            if request.method == 'GET':
                return jsonify({
                    'max_memory_usage_gb': self.controller.config.max_memory_usage_gb,
                    'max_cpu_usage_percent': self.controller.config.max_cpu_usage_percent,
                    'max_gpu_usage_percent': self.controller.config.max_gpu_usage_percent,
                    'auto_cleanup_enabled': self.controller.config.auto_cleanup_enabled
                })
            else:
                try:
                    new_config = request.json
                    self.controller.update_configuration(new_config)
                    return jsonify({'success': True, 'message': 'Configuration updated'})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
                    
        @self.app.route('/api/processor-stats')
        def get_processor_stats():
            """Get processor statistics"""
            try:
                if self.processor:
                    stats = self.processor.get_mining_statistics()
                    return jsonify(stats)
                else:
                    return jsonify({'error': 'Processor not available'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/save-config', methods=['POST'])
        def save_config():
            """Save current configuration to file"""
            try:
                self.controller.save_configuration()
                return jsonify({'success': True, 'message': 'Configuration saved'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/load-config', methods=['POST'])
        def load_config():
            """Load configuration from file"""
            try:
                self.controller.load_configuration()
                return jsonify({'success': True, 'message': 'Configuration loaded'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/start-monitoring', methods=['POST'])
        def start_monitoring():
            """Start system monitoring"""
            try:
                if not self.controller.is_monitoring:
                    # Start monitoring in background thread
                    def run_monitoring():
                        asyncio.run(self.controller.start_monitoring())
                    
                    self.background_thread = threading.Thread(target=run_monitoring)
                    self.background_thread.daemon = True
                    self.background_thread.start()
                    
                return jsonify({'success': True, 'message': 'Monitoring started'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/stop-monitoring', methods=['POST'])
        def stop_monitoring():
            """Stop system monitoring"""
            try:
                asyncio.run(self.controller.stop_monitoring())
                return jsonify({'success': True, 'message': 'Monitoring stopped'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
    def create_template(self):
        """Create the HTML template for the dashboard"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Processor Control Dashboard</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .panel h2 {
            margin-top: 0;
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .metric {
            background-color: #3d3d3d;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #ffffff;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #cccccc;
            margin-top: 5px;
        }
        
        .feature-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .feature-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #3d3d3d;
            border-radius: 6px;
            margin-bottom: 5px;
        }
        
        .toggle-switch {
            position: relative;
            width: 60px;
            height: 30px;
            background-color: #ccc;
            border-radius: 15px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .toggle-switch.active {
            background-color: #4CAF50;
        }
        
        .toggle-slider {
            position: absolute;
            top: 3px;
            left: 3px;
            width: 24px;
            height: 24px;
            background-color: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        
        .toggle-switch.active .toggle-slider {
            transform: translateX(30px);
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.3s;
            margin: 5px;
        }
        
        .btn-primary {
            background-color: #4CAF50;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #45a049;
        }
        
        .btn-danger {
            background-color: #f44336;
            color: white;
        }
        
        .btn-danger:hover {
            background-color: #da190b;
        }
        
        .btn-warning {
            background-color: #ff9800;
            color: white;
        }
        
        .btn-warning:hover {
            background-color: #e68900;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-green {
            background-color: #4CAF50;
        }
        
        .status-red {
            background-color: #f44336;
        }
        
        .status-yellow {
            background-color: #ff9800;
        }
        
        .threshold-input {
            width: 80px;
            padding: 5px;
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #4d4d4d;
            color: #ffffff;
            margin-left: 10px;
        }
        
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .alert-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .log-container {
            background-color: #000000;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            height: 200px;
            overflow-y: auto;
            border: 1px solid #333;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #3d3d3d;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            transition: width 0.3s ease;
        }
        
        .emergency-panel {
            background-color: #4d1f1f;
            border: 2px solid #f44336;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .feature-controls {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚀 BTC Processor Control Dashboard</h1>
            <p>Real-time monitoring and control for Bitcoin data processing</p>
        </div>
        
        <div id="alerts"></div>
        
        <div class="grid">
            <!-- System Metrics Panel -->
            <div class="panel">
                <h2>📊 System Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="cpu-usage">--</div>
                        <div class="metric-label">CPU Usage (%)</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="cpu-progress"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="memory-usage">--</div>
                        <div class="metric-label">Memory (GB)</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="memory-progress"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="gpu-usage">--</div>
                        <div class="metric-label">GPU Usage (%)</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="gpu-progress"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="disk-usage">--</div>
                        <div class="metric-label">Disk (GB)</div>
                    </div>
                </div>
                
                <div style="margin-top: 15px;">
                    <span class="status-indicator" id="monitoring-status"></span>
                    <span id="monitoring-text">Monitoring Status</span>
                </div>
            </div>
            
            <!-- Feature Controls Panel -->
            <div class="panel">
                <h2>⚙️ Feature Controls</h2>
                <div class="feature-controls">
                    <div class="feature-item">
                        <span>Mining Analysis</span>
                        <div class="toggle-switch" data-feature="mining_analysis">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Block Timing</span>
                        <div class="toggle-switch" data-feature="block_timing">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Nonce Sequences</span>
                        <div class="toggle-switch" data-feature="nonce_sequences">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Difficulty Tracking</span>
                        <div class="toggle-switch" data-feature="difficulty_tracking">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Memory Management</span>
                        <div class="toggle-switch" data-feature="memory_management">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Hash Generation</span>
                        <div class="toggle-switch" data-feature="hash_generation">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Load Balancing</span>
                        <div class="toggle-switch" data-feature="load_balancing">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span>Storage</span>
                        <div class="toggle-switch" data-feature="storage">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <button class="btn btn-primary" onclick="enableAllAnalysis()">Enable All Analysis</button>
                    <button class="btn btn-warning" onclick="disableAllAnalysis()">Disable All Analysis</button>
                </div>
            </div>
            
            <!-- System Thresholds Panel -->
            <div class="panel">
                <h2>⚠️ System Thresholds</h2>
                <div style="margin: 10px 0;">
                    <label>Memory Warning (GB):</label>
                    <input type="number" class="threshold-input" id="memory-warning" step="0.5" min="1" max="50">
                </div>
                <div style="margin: 10px 0;">
                    <label>Memory Critical (GB):</label>
                    <input type="number" class="threshold-input" id="memory-critical" step="0.5" min="1" max="50">
                </div>
                <div style="margin: 10px 0;">
                    <label>CPU Warning (%):</label>
                    <input type="number" class="threshold-input" id="cpu-warning" min="1" max="100">
                </div>
                <div style="margin: 10px 0;">
                    <label>CPU Critical (%):</label>
                    <input type="number" class="threshold-input" id="cpu-critical" min="1" max="100">
                </div>
                <div style="margin: 10px 0;">
                    <label>GPU Warning (%):</label>
                    <input type="number" class="threshold-input" id="gpu-warning" min="1" max="100">
                </div>
                <div style="margin: 10px 0;">
                    <label>GPU Critical (%):</label>
                    <input type="number" class="threshold-input" id="gpu-critical" min="1" max="100">
                </div>
                
                <button class="btn btn-primary" onclick="updateThresholds()">Update Thresholds</button>
            </div>
            
            <!-- Emergency Controls Panel -->
            <div class="panel emergency-panel">
                <h2>🚨 Emergency Controls</h2>
                <p>Use these controls when system resources are critically low</p>
                
                <button class="btn btn-danger" onclick="emergencyCleanup()">Emergency Memory Cleanup</button>
                <button class="btn btn-warning" onclick="stopMonitoring()">Stop Monitoring</button>
                
                <div style="margin-top: 15px;">
                    <div class="alert" id="emergency-status" style="display: none;"></div>
                </div>
            </div>
            
            <!-- Configuration Panel -->
            <div class="panel">
                <h2>🔧 Configuration</h2>
                <div style="margin: 10px 0;">
                    <label>Max Memory Usage (GB):</label>
                    <input type="number" class="threshold-input" id="max-memory" step="0.5" min="1" max="100">
                </div>
                <div style="margin: 10px 0;">
                    <label>Max CPU Usage (%):</label>
                    <input type="number" class="threshold-input" id="max-cpu" min="1" max="100">
                </div>
                <div style="margin: 10px 0;">
                    <label>Max GPU Usage (%):</label>
                    <input type="number" class="threshold-input" id="max-gpu" min="1" max="100">
                </div>
                
                <div style="margin-top: 15px;">
                    <button class="btn btn-primary" onclick="updateConfiguration()">Update Config</button>
                    <button class="btn btn-primary" onclick="saveConfiguration()">Save Config</button>
                    <button class="btn btn-primary" onclick="loadConfiguration()">Load Config</button>
                </div>
            </div>
            
            <!-- Status Log Panel -->
            <div class="panel">
                <h2>📋 Status Log</h2>
                <div class="log-container" id="status-log">
                    <div>System initialized...</div>
                </div>
                <button class="btn btn-primary" onclick="clearLog()" style="margin-top: 10px;">Clear Log</button>
            </div>
        </div>
        
        <!-- Control Buttons -->
        <div style="text-align: center; margin-top: 20px;">
            <button class="btn btn-primary" onclick="startMonitoring()">Start Monitoring</button>
            <button class="btn btn-warning" onclick="stopMonitoring()">Stop Monitoring</button>
            <button class="btn btn-primary" onclick="refreshStatus()">Refresh Status</button>
        </div>
    </div>
    
    <script>
        let refreshInterval;
        
        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeToggleSwitches();
            refreshStatus();
            startAutoRefresh();
        });
        
        function initializeToggleSwitches() {
            document.querySelectorAll('.toggle-switch').forEach(toggle => {
                toggle.addEventListener('click', function() {
                    const feature = this.dataset.feature;
                    const isActive = this.classList.contains('active');
                    
                    if (isActive) {
                        disableFeature(feature);
                    } else {
                        enableFeature(feature);
                    }
                });
            });
        }
        
        function enableFeature(featureName) {
            fetch(`/api/features/${featureName}/enable`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage(`✅ Enabled ${featureName}`, 'success');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error enabling ${featureName}: ${data.error}`, 'error');
                    }
                })
                .catch(error => {
                    logMessage(`❌ Network error: ${error}`, 'error');
                });
        }
        
        function disableFeature(featureName) {
            fetch(`/api/features/${featureName}/disable`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage(`⏹️ Disabled ${featureName}`, 'warning');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error disabling ${featureName}: ${data.error}`, 'error');
                    }
                })
                .catch(error => {
                    logMessage(`❌ Network error: ${error}`, 'error');
                });
        }
        
        function enableAllAnalysis() {
            fetch('/api/features/analysis/enable-all', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('✅ Enabled all analysis features', 'success');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error: ${data.error}`, 'error');
                    }
                });
        }
        
        function disableAllAnalysis() {
            fetch('/api/features/analysis/disable-all', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('⏹️ Disabled all analysis features', 'warning');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error: ${data.error}`, 'error');
                    }
                });
        }
        
        function emergencyCleanup() {
            if (confirm('Are you sure you want to perform emergency cleanup? This will clear memory buffers and disable non-essential features.')) {
                fetch('/api/emergency-cleanup', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            logMessage('🚨 Emergency cleanup completed', 'warning');
                            showAlert('Emergency cleanup completed successfully', 'success');
                            refreshStatus();
                        } else {
                            logMessage(`❌ Emergency cleanup error: ${data.error}`, 'error');
                            showAlert('Emergency cleanup failed', 'danger');
                        }
                    });
            }
        }
        
        function startMonitoring() {
            fetch('/api/start-monitoring', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('▶️ System monitoring started', 'success');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error starting monitoring: ${data.error}`, 'error');
                    }
                });
        }
        
        function stopMonitoring() {
            fetch('/api/stop-monitoring', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('⏹️ System monitoring stopped', 'warning');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error stopping monitoring: ${data.error}`, 'error');
                    }
                });
        }
        
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateSystemMetrics(data.system_metrics);
                    updateFeatureStates(data.feature_states);
                    updateMonitoringStatus(data.monitoring_active);
                    updateThresholdInputs(data.thresholds);
                    updateConfigurationInputs(data.configuration);
                })
                .catch(error => {
                    logMessage(`❌ Status refresh error: ${error}`, 'error');
                });
        }
        
        function updateSystemMetrics(metrics) {
            document.getElementById('cpu-usage').textContent = metrics.cpu_usage.toFixed(1);
            document.getElementById('memory-usage').textContent = metrics.memory_usage_gb.toFixed(1);
            document.getElementById('gpu-usage').textContent = metrics.gpu_usage.toFixed(1);
            document.getElementById('disk-usage').textContent = metrics.disk_usage_gb.toFixed(1);
            
            // Update progress bars
            document.getElementById('cpu-progress').style.width = `${Math.min(metrics.cpu_usage, 100)}%`;
            document.getElementById('memory-progress').style.width = `${Math.min((metrics.memory_usage_gb / 16) * 100, 100)}%`;
            document.getElementById('gpu-progress').style.width = `${Math.min(metrics.gpu_usage, 100)}%`;
        }
        
        function updateFeatureStates(featureStates) {
            Object.entries(featureStates).forEach(([feature, enabled]) => {
                const toggle = document.querySelector(`[data-feature="${feature}"]`);
                if (toggle) {
                    if (enabled) {
                        toggle.classList.add('active');
                    } else {
                        toggle.classList.remove('active');
                    }
                }
            });
        }
        
        function updateMonitoringStatus(isActive) {
            const statusIndicator = document.getElementById('monitoring-status');
            const statusText = document.getElementById('monitoring-text');
            
            if (isActive) {
                statusIndicator.className = 'status-indicator status-green';
                statusText.textContent = 'Monitoring Active';
            } else {
                statusIndicator.className = 'status-indicator status-red';
                statusText.textContent = 'Monitoring Inactive';
            }
        }
        
        function updateThresholdInputs(thresholds) {
            document.getElementById('memory-warning').value = thresholds.memory_warning;
            document.getElementById('memory-critical').value = thresholds.memory_critical;
            document.getElementById('cpu-warning').value = thresholds.cpu_warning;
            document.getElementById('cpu-critical').value = thresholds.cpu_critical;
            document.getElementById('gpu-warning').value = thresholds.gpu_warning;
            document.getElementById('gpu-critical').value = thresholds.gpu_critical;
        }
        
        function updateConfigurationInputs(config) {
            document.getElementById('max-memory').value = config.max_memory_gb;
            document.getElementById('max-cpu').value = config.max_cpu_percent;
            document.getElementById('max-gpu').value = config.max_gpu_percent;
        }
        
        function updateThresholds() {
            const thresholds = {
                memory_warning: parseFloat(document.getElementById('memory-warning').value),
                memory_critical: parseFloat(document.getElementById('memory-critical').value),
                cpu_warning: parseFloat(document.getElementById('cpu-warning').value),
                cpu_critical: parseFloat(document.getElementById('cpu-critical').value),
                gpu_warning: parseFloat(document.getElementById('gpu-warning').value),
                gpu_critical: parseFloat(document.getElementById('gpu-critical').value)
            };
            
            fetch('/api/thresholds', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(thresholds)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    logMessage('✅ Thresholds updated', 'success');
                } else {
                    logMessage(`❌ Error updating thresholds: ${data.error}`, 'error');
                }
            });
        }
        
        function updateConfiguration() {
            const config = {
                max_memory_usage_gb: parseFloat(document.getElementById('max-memory').value),
                max_cpu_usage_percent: parseFloat(document.getElementById('max-cpu').value),
                max_gpu_usage_percent: parseFloat(document.getElementById('max-gpu').value)
            };
            
            fetch('/api/configuration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    logMessage('✅ Configuration updated', 'success');
                } else {
                    logMessage(`❌ Error updating configuration: ${data.error}`, 'error');
                }
            });
        }
        
        function saveConfiguration() {
            fetch('/api/save-config', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('💾 Configuration saved', 'success');
                    } else {
                        logMessage(`❌ Error saving configuration: ${data.error}`, 'error');
                    }
                });
        }
        
        function loadConfiguration() {
            fetch('/api/load-config', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logMessage('📁 Configuration loaded', 'success');
                        refreshStatus();
                    } else {
                        logMessage(`❌ Error loading configuration: ${data.error}`, 'error');
                    }
                });
        }
        
        function logMessage(message, type = 'info') {
            const logContainer = document.getElementById('status-log');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.textContent = `[${timestamp}] ${message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        function clearLog() {
            document.getElementById('status-log').innerHTML = '<div>Log cleared...</div>';
        }
        
        function showAlert(message, type) {
            const alertsContainer = document.getElementById('alerts');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            alertsContainer.appendChild(alert);
            
            setTimeout(() => {
                alert.remove();
            }, 5000);
        }
        
        function startAutoRefresh() {
            refreshInterval = setInterval(refreshStatus, 5000); // Refresh every 5 seconds
        }
        
        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                stopAutoRefresh();
            } else {
                startAutoRefresh();
            }
        });
    </script>
</body>
</html>
        """
        
        # Ensure templates directory exists
        from pathlib import Path
        templates_dir = Path('templates')
        templates_dir.mkdir(exist_ok=True)
        
        # Write template file
        template_file = templates_dir / 'btc_processor_dashboard.html'
        with open(template_file, 'w') as f:
            f.write(template_content)
            
    def run(self, debug=False):
        """Run the web UI server"""
        try:
            self.create_template()
            logger.info(f"Starting BTC Processor UI on {self.host}:{self.port}")
            self.is_running = True
            self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"UI server error: {e}")
            raise
            
    def stop(self):
        """Stop the web UI server"""
        self.is_running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)

def create_ui_instance(processor=None, host='localhost', port=5000):
    """Create a UI instance for the BTC processor"""
    return BTCProcessorUI(processor, host, port)

if __name__ == "__main__":
    # Example usage
    ui = BTCProcessorUI()
    ui.run(debug=True) 