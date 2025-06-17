#!/usr/bin/env python3
"""
React Dashboard Integration
===========================

Serves the React dashboard (practical_schwabot_dashboard.tsx) with real-time data
from the Schwabot visual integration bridge and sustainment systems.

This module provides:
- Flask web server for serving React dashboard
- WebSocket endpoints for real-time data streaming
- API endpoints for configuration and control
- Integration with all backend systems
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Core system imports
from .visual_integration_bridge import VisualIntegrationBridge, VisualMetrics, PatternState
from .ui_state_bridge import UIStateBridge, SystemStatus
from .sustainment_underlay_controller import SustainmentUnderlayController

logger = logging.getLogger(__name__)

class ReactDashboardServer:
    """
    Flask server that serves the React dashboard and provides real-time data APIs.
    
    This integrates the practical_schwabot_dashboard.tsx component with the
    backend Schwabot systems through WebSocket and REST APIs.
    """
    
    def __init__(self,
                 visual_bridge: VisualIntegrationBridge,
                 ui_bridge: UIStateBridge,
                 sustainment_controller: SustainmentUnderlayController,
                 host: str = "localhost",
                 port: int = 5000):
        """
        Initialize React dashboard server
        
        Args:
            visual_bridge: Visual integration bridge for real-time data
            ui_bridge: UI state bridge for system data
            sustainment_controller: Sustainment underlay controller
            host: Flask server host
            port: Flask server port
        """
        
        # Core components
        self.visual_bridge = visual_bridge
        self.ui_bridge = ui_bridge
        self.sustainment_controller = sustainment_controller
        
        # Flask app setup
        self.app = Flask(__name__, static_folder='../static', template_folder='../templates')
        self.app.config['SECRET_KEY'] = 'schwabot_dashboard_secret_key'
        
        # Enable CORS for development
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        # SocketIO for real-time communication
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Server configuration
        self.host = host
        self.port = port
        self.server_active = False
        self.server_thread = None
        
        # Data streaming
        self.streaming_active = False
        self.streaming_thread = None
        self.connected_clients = set()
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        logger.info(f"React Dashboard Server initialized on {host}:{port}")

    def _setup_routes(self) -> None:
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Serve the main dashboard page"""
            return self._render_dashboard_html()
        
        @self.app.route('/api/status')
        def api_status():
            """Get current system status"""
            try:
                ui_state = self.ui_bridge.get_ui_state()
                sustainment_status = self.sustainment_controller.get_sustainment_status()
                tesseract_status = self.visual_bridge.get_tesseract_status()
                
                return jsonify({
                    'status': 'ok',
                    'timestamp': datetime.now().isoformat(),
                    'system_health': ui_state.get('system_health', {}),
                    'sustainment_status': sustainment_status,
                    'tesseract_status': tesseract_status,
                    'connected_clients': len(self.connected_clients)
                })
                
            except Exception as e:
                logger.error(f"API status error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """Get current visual metrics"""
            try:
                if self.visual_bridge.current_metrics:
                    return jsonify({
                        'status': 'ok',
                        'metrics': self.visual_bridge.current_metrics.__dict__,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'no_data',
                        'message': 'No metrics available'
                    })
                    
            except Exception as e:
                logger.error(f"API metrics error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/pattern_state')
        def api_pattern_state():
            """Get current pattern analysis state"""
            try:
                if self.visual_bridge.current_pattern_state:
                    return jsonify({
                        'status': 'ok',
                        'pattern_state': self.visual_bridge.current_pattern_state.__dict__,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'no_data',
                        'message': 'No pattern state available'
                    })
                    
            except Exception as e:
                logger.error(f"API pattern state error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """Get or update system configuration"""
            if request.method == 'GET':
                # Return current configuration
                return jsonify({
                    'status': 'ok',
                    'config': {
                        'update_interval': 0.1,
                        'websocket_port': self.visual_bridge.websocket_port,
                        'sustainment_threshold': self.sustainment_controller.s_crit,
                        'tesseract_enabled': self.visual_bridge.get_tesseract_status()['tesseract_available']
                    }
                })
            else:
                # Update configuration
                try:
                    config = request.get_json()
                    # Apply configuration changes...
                    logger.info(f"Configuration updated: {config}")
                    return jsonify({'status': 'ok', 'message': 'Configuration updated'})
                except Exception as e:
                    logger.error(f"Configuration update error: {e}")
                    return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/export')
        def api_export():
            """Export historical data"""
            try:
                export_data = self.visual_bridge._prepare_export_data()
                return jsonify({
                    'status': 'ok',
                    'data': export_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"API export error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

    def _setup_socket_handlers(self) -> None:
        """Setup SocketIO handlers for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            logger.info(f"Client connected: {client_id}")
            
            # Send initial data
            self._send_initial_data(client_id)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            logger.info(f"Client disconnected: {client_id}")
        
        @self.socketio.on('request_data')
        def handle_request_data(data):
            """Handle data requests from clients"""
            try:
                data_type = data.get('type', 'all')
                self._send_requested_data(request.sid, data_type)
            except Exception as e:
                logger.error(f"Error handling data request: {e}")
        
        @self.socketio.on('configure_system')
        def handle_configure_system(data):
            """Handle system configuration requests"""
            try:
                config = data.get('config', {})
                # Apply configuration...
                emit('configuration_applied', {'status': 'ok'})
            except Exception as e:
                logger.error(f"Error handling configuration: {e}")
                emit('configuration_applied', {'status': 'error', 'message': str(e)})

    def _render_dashboard_html(self) -> str:
        """Render the HTML page that loads the React dashboard"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schwabot Trading System</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/recharts@2.8.0/esm/index.js" type="module"></script>
    <script src="https://unpkg.com/lucide-react@0.263.1/dist/umd/lucide-react.js"></script>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { margin: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    </style>
</head>
<body>
    <div id="dashboard-root"></div>
    
    <script type="module">
        // Initialize Socket.IO connection
        const socket = io();
        
        // Global state for the dashboard
        window.schwabotState = {
            connected: false,
            lastUpdate: null,
            systemHealth: {},
            patternState: {},
            metrics: {}
        };
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to Schwabot server');
            window.schwabotState.connected = true;
            socket.emit('request_data', { type: 'all' });
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            window.schwabotState.connected = false;
        });
        
        socket.on('system_update', (data) => {
            window.schwabotState.lastUpdate = new Date();
            if (data.system_health) window.schwabotState.systemHealth = data.system_health;
            if (data.pattern_state) window.schwabotState.patternState = data.pattern_state;
            if (data.metrics) window.schwabotState.metrics = data.metrics;
            
            // Trigger React re-render if component is mounted
            if (window.schwabotDashboard) {
                window.schwabotDashboard.forceUpdate();
            }
        });
        
        // Load and render the React dashboard component
        // Note: In production, this would be a compiled React bundle
        // For now, we'll create a simplified version that connects to our data
        
        const { useState, useEffect } = React;
        
        function SchwabotDashboard() {
            const [state, setState] = useState(window.schwabotState);
            
            useEffect(() => {
                // Set up periodic state updates
                const interval = setInterval(() => {
                    setState({...window.schwabotState});
                }, 1000);
                
                // Store reference for forced updates
                window.schwabotDashboard = { forceUpdate: () => setState({...window.schwabotState}) };
                
                return () => {
                    clearInterval(interval);
                    delete window.schwabotDashboard;
                };
            }, []);
            
            return React.createElement('div', {
                className: 'min-h-screen bg-gray-900 text-white p-6'
            }, [
                // Header
                React.createElement('div', { key: 'header', className: 'mb-8' }, [
                    React.createElement('div', { 
                        key: 'header-content',
                        className: 'flex justify-between items-center' 
                    }, [
                        React.createElement('h1', {
                            key: 'title',
                            className: 'text-3xl font-bold text-blue-400'
                        }, 'Schwabot Trading System'),
                        React.createElement('div', {
                            key: 'status',
                            className: 'text-right'
                        }, [
                            React.createElement('div', {
                                key: 'time',
                                className: 'text-lg font-mono'
                            }, new Date().toLocaleTimeString()),
                            React.createElement('div', {
                                key: 'connection',
                                className: `text-sm ${state.connected ? 'text-green-400' : 'text-red-400'}`
                            }, state.connected ? 'System Operational' : 'Disconnected')
                        ])
                    ])
                ]),
                
                // Status message
                React.createElement('div', {
                    key: 'status-msg',
                    className: 'text-center py-8'
                }, [
                    React.createElement('h2', {
                        key: 'msg-title',
                        className: 'text-xl text-blue-400 mb-4'
                    }, 'React Dashboard Integration Active'),
                    React.createElement('p', {
                        key: 'msg-content',
                        className: 'text-gray-300'
                    }, `Connected: ${state.connected ? 'Yes' : 'No'} | Last Update: ${state.lastUpdate ? state.lastUpdate.toLocaleTimeString() : 'Never'}`),
                    React.createElement('div', {
                        key: 'live-data',
                        className: 'mt-4 p-4 bg-gray-800 rounded-lg max-w-md mx-auto'
                    }, [
                        React.createElement('h3', {
                            key: 'data-title',
                            className: 'text-lg text-green-400 mb-2'
                        }, 'Live Data Stream'),
                        React.createElement('pre', {
                            key: 'data-content',
                            className: 'text-xs text-gray-300 overflow-auto'
                        }, JSON.stringify({
                            sustainmentIndex: state.systemHealth.sustainment_index || 'N/A',
                            profit24h: state.systemHealth.profit_24h || 'N/A',
                            activePatterns: state.patternState.active_patterns?.length || 0,
                            lastMetrics: state.metrics.timestamp || 'N/A'
                        }, null, 2))
                    ])
                ])
            ]);
        }
        
        // Render the dashboard
        const root = ReactDOM.createRoot(document.getElementById('dashboard-root'));
        root.render(React.createElement(SchwabotDashboard));
        
        console.log('Schwabot React Dashboard initialized');
    </script>
</body>
</html>
        """
        
        return html_template

    def _send_initial_data(self, client_id: str) -> None:
        """Send initial data to a newly connected client"""
        
        try:
            # Get current system state
            ui_state = self.ui_bridge.get_ui_state()
            pattern_state = self.visual_bridge.current_pattern_state
            metrics = self.visual_bridge.current_metrics
            
            # Send initial data package
            self.socketio.emit('system_update', {
                'type': 'initial_data',
                'system_health': ui_state.get('system_health', {}),
                'pattern_state': pattern_state.__dict__ if pattern_state else {},
                'metrics': metrics.__dict__ if metrics else {},
                'timestamp': datetime.now().isoformat()
            }, room=client_id)
            
        except Exception as e:
            logger.error(f"Error sending initial data to client {client_id}: {e}")

    def _send_requested_data(self, client_id: str, data_type: str) -> None:
        """Send requested data to a specific client"""
        
        try:
            data = {}
            
            if data_type in ['all', 'system_health']:
                ui_state = self.ui_bridge.get_ui_state()
                data['system_health'] = ui_state.get('system_health', {})
            
            if data_type in ['all', 'pattern_state']:
                if self.visual_bridge.current_pattern_state:
                    data['pattern_state'] = self.visual_bridge.current_pattern_state.__dict__
            
            if data_type in ['all', 'metrics']:
                if self.visual_bridge.current_metrics:
                    data['metrics'] = self.visual_bridge.current_metrics.__dict__
            
            # Send the requested data
            self.socketio.emit('system_update', {
                'type': 'requested_data',
                'data_type': data_type,
                **data,
                'timestamp': datetime.now().isoformat()
            }, room=client_id)
            
        except Exception as e:
            logger.error(f"Error sending requested data to client {client_id}: {e}")

    def start_streaming(self) -> None:
        """Start real-time data streaming to connected clients"""
        
        if self.streaming_active:
            logger.warning("Data streaming already active")
            return
        
        self.streaming_active = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True
        )
        self.streaming_thread.start()
        
        logger.info("Real-time data streaming started")

    def _streaming_loop(self) -> None:
        """Main streaming loop for real-time data updates"""
        
        while self.streaming_active:
            try:
                if self.connected_clients:
                    # Get current data
                    ui_state = self.ui_bridge.get_ui_state()
                    pattern_state = self.visual_bridge.current_pattern_state
                    metrics = self.visual_bridge.current_metrics
                    
                    # Broadcast update to all connected clients
                    self.socketio.emit('system_update', {
                        'type': 'live_update',
                        'system_health': ui_state.get('system_health', {}),
                        'pattern_state': pattern_state.__dict__ if pattern_state else {},
                        'metrics': metrics.__dict__ if metrics else {},
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(1.0)

    def start_server(self, debug: bool = False) -> None:
        """Start the Flask server"""
        
        if self.server_active:
            logger.warning("Server already running")
            return
        
        self.server_active = True
        
        # Start data streaming
        self.start_streaming()
        
        # Start Flask server in thread for non-blocking operation
        def run_server():
            try:
                self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=debug,
                    use_reloader=False,
                    log_output=not debug
                )
            except Exception as e:
                logger.error(f"Flask server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"React Dashboard Server started on http://{self.host}:{self.port}")

    def stop_server(self) -> None:
        """Stop the Flask server and data streaming"""
        
        self.server_active = False
        self.streaming_active = False
        
        # Stop streaming thread
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5.0)
        
        # Note: Flask server thread will stop when the main process ends
        
        logger.info("React Dashboard Server stopped")

# Factory function for easy integration
def create_react_dashboard(visual_bridge: VisualIntegrationBridge,
                          ui_bridge: UIStateBridge,
                          sustainment_controller: SustainmentUnderlayController,
                          port: int = 5000) -> ReactDashboardServer:
    """Factory function to create React dashboard server"""
    
    server = ReactDashboardServer(
        visual_bridge=visual_bridge,
        ui_bridge=ui_bridge,
        sustainment_controller=sustainment_controller,
        port=port
    )
    
    # Start the server
    server.start_server()
    
    logger.info(f"React Dashboard Server created and started on port {port}")
    return server 