#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 DYNAMIC TIMING WEB DASHBOARD - HUMAN-FRIENDLY INTERFACE
==========================================================

Advanced web-based dashboard for the dynamic timing system:
- Real-time performance monitoring
- Interactive charts and visualizations
- Human-friendly trading recommendations
- System status and health monitoring
- Mobile-responsive design
- INTEGRATED CONTROL CENTER - Advanced Options & Visual Controls

Features:
- WebSocket real-time updates
- Interactive charts with Plotly
- Trading recommendations engine
- System health monitoring
- User-friendly explanations
- ADVANCED OPTIONS INTEGRATION
- VISUAL CONTROLS INTEGRATION
- USB MANAGEMENT INTEGRATION
- API KEY MANAGEMENT INTEGRATION
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import subprocess
import sys
import os

# Import our dynamic timing systems
try:
    from core.dynamic_timing_system import get_dynamic_timing_system, TimingRegime, OrderTiming
    from core.enhanced_real_time_data_puller import get_enhanced_data_puller, DataSource
    DYNAMIC_TIMING_AVAILABLE = True
except ImportError:
    DYNAMIC_TIMING_AVAILABLE = False
    print("Warning: Dynamic timing system not available")

# Add imports for 2025 performance optimizer
try:
    from core.performance_optimizer_2025 import performance_optimizer
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    print("⚠️ Performance optimizer not available")

# Add to the DynamicTimingDashboard class initialization
def __init__(self):
    """Initialize the dynamic timing dashboard."""
    self.dashboard_data = {
        'system_status': 'initializing',
        'current_regime': 'neutral',
        'rolling_profit': 0.0,
        'timing_accuracy': 0.0,
        'total_signals': 0,
        'successful_signals': 0,
        'usb_status': 'unknown',
        'api_keys_configured': False,
        'compression_active': False,
        'visual_controls_active': False,
        'advanced_options_available': False,
        'performance_optimization_active': False,
        'optimization_level': 'unknown',
        'acceleration_type': 'unknown'
    }
    
    # Initialize 2025 performance optimizer
    if PERFORMANCE_OPTIMIZER_AVAILABLE:
        try:
            # Get hardware info and optimize
            from core.hardware_auto_detector import hardware_detector
            hw_info = hardware_detector.detect_hardware()
            hw_dict = {
                'ram_gb': hw_info.ram_gb,
                'cpu_cores': hw_info.cpu_cores,
                'gpu_memory_gb': hw_info.gpu.memory_gb if hasattr(hw_info, 'gpu') else 0.0
            }
            
            # Detect and apply optimal profile
            optimal_profile = performance_optimizer.detect_optimal_profile(hw_dict)
            optimizations = performance_optimizer.optimize_system(optimal_profile)
            
            # Update dashboard data
            self.dashboard_data.update({
                'performance_optimization_active': True,
                'optimization_level': optimal_profile.optimization_level.value,
                'acceleration_type': optimal_profile.acceleration_type.value,
                'max_concurrent_trades': optimal_profile.max_concurrent_trades,
                'max_charts_per_device': optimal_profile.max_charts_per_device,
                'data_processing_latency_ms': optimal_profile.data_processing_latency_ms
            })
            
            print(f"✅ 2025 Performance Optimization: {optimal_profile.optimization_level.value} with {optimal_profile.acceleration_type.value}")
        except Exception as e:
            print(f"❌ Performance optimization failed: {e}")
    
    # Initialize other components
    try:
        from core.storage_device_manager import StorageDeviceManager
        self.storage_manager = StorageDeviceManager()
    except ImportError:
        self.storage_manager = None
        print("⚠️ Storage device manager not available")
    
    try:
        from gui.visual_controls_gui import VisualControlsGUI
        self.visual_controls = VisualControlsGUI()
    except ImportError:
        self.visual_controls = None
        print("⚠️ Visual controls not available")
    
    self.advanced_controls_data = {
        'usb_devices': [],
        'storage_info': {},
        'api_keys': {},
        'visual_settings': {},
        'compression_stats': {},
        'performance_metrics': {}
    }

# Import advanced control systems
try:
    from alpha_compression_manager import (
        get_storage_device_manager, 
        StorageDevice, 
        AlphaCompressionManager,
        compress_trading_data_on_device,
        auto_compress_device_data,
        get_device_compression_suggestions
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    print("Warning: Alpha compression system not available")

try:
    from usb_manager import auto_detect_usb, setup_usb_storage, get_usb_status
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("Warning: USB management system not available")

try:
    from api_key_manager import get_api_key_status, get_configured_exchanges
    API_KEYS_AVAILABLE = True
except ImportError:
    API_KEYS_AVAILABLE = False
    print("Warning: API key management system not available")

try:
    from visual_controls_gui import VisualControlsGUI
    VISUAL_CONTROLS_AVAILABLE = True
except ImportError:
    VISUAL_CONTROLS_AVAILABLE = False
    print("Warning: Visual controls system not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'schwabot_dynamic_timing_dashboard_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

class DynamicTimingDashboard:
    """Web dashboard for dynamic timing system with integrated control center."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.dynamic_timing = get_dynamic_timing_system() if DYNAMIC_TIMING_AVAILABLE else None
        self.data_puller = get_enhanced_data_puller() if DYNAMIC_TIMING_AVAILABLE else None
        
        # Initialize advanced control systems
        self.device_manager = get_storage_device_manager() if COMPRESSION_AVAILABLE else None
        self.visual_controls = VisualControlsGUI() if VISUAL_CONTROLS_AVAILABLE else None
        
        # Dashboard state
        self.dashboard_data = {
            'system_status': 'initializing',
            'current_regime': 'normal',
            'rolling_profit': 0.0,
            'timing_accuracy': 0.0,
            'total_signals': 0,
            'successful_signals': 0,
            'system_uptime': 0,
            'current_volatility': 0.0,
            'current_momentum': 0.0,
            'last_update': datetime.now().isoformat(),
            # Advanced control status
            'usb_status': 'unknown',
            'api_keys_configured': False,
            'compression_active': False,
            'visual_controls_active': False,
            'advanced_options_available': True
        }
        
        # Historical data for charts
        self.historical_data = {
            'profit': [],
            'volatility': [],
            'momentum': [],
            'regime_changes': [],
            'timing_events': []
        }
        
        # Trading recommendations
        self.recommendations = []
        
        # Advanced control data
        self.advanced_controls_data = {
            'usb_devices': [],
            'storage_devices': [],
            'api_keys': [],
            'visual_settings': {},
            'compression_stats': {}
        }
        
        # Setup routes
        self.setup_routes()
        
        # Start data collection
        self.start_data_collection()
    
    def setup_routes(self):
        """Setup Flask routes with integrated control center."""
        
        @app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('dynamic_timing_dashboard.html')
        
        @app.route('/api/status')
        def get_status():
            """Get current system status."""
            return jsonify(self.dashboard_data)
        
        @app.route('/api/historical')
        def get_historical():
            """Get historical data for charts."""
            return jsonify(self.historical_data)
        
        @app.route('/api/recommendations')
        def get_recommendations():
            """Get trading recommendations."""
            return jsonify(self.recommendations)
        
        # === ADVANCED CONTROL CENTER ENDPOINTS ===
        
        @app.route('/api/advanced/usb/status')
        def get_usb_status_api():
            """Get USB device status."""
            try:
                if USB_AVAILABLE:
                    status = get_usb_status()
                    self.dashboard_data['usb_status'] = status.get('status', 'unknown')
                    return jsonify(status)
                else:
                    return jsonify({'error': 'USB management not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/usb/detect', methods=['POST'])
        def detect_usb_devices():
            """Detect USB devices."""
            try:
                if USB_AVAILABLE:
                    devices = auto_detect_usb()
                    self.advanced_controls_data['usb_devices'] = devices
                    return jsonify({'status': 'success', 'devices': devices})
                else:
                    return jsonify({'error': 'USB management not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/usb/setup', methods=['POST'])
        def setup_usb_storage_api():
            """Setup USB storage."""
            try:
                if USB_AVAILABLE:
                    result = setup_usb_storage()
                    return jsonify({'status': 'success', 'result': result})
                else:
                    return jsonify({'error': 'USB management not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/compression/status')
        def get_compression_status():
            """Get compression system status."""
            try:
                if COMPRESSION_AVAILABLE and self.device_manager:
                    devices = self.device_manager.get_all_devices()
                    stats = {}
                    for device in devices:
                        stats[device.name] = {
                            'compressed_size': device.get_compressed_size(),
                            'original_size': device.get_original_size(),
                            'compression_ratio': device.get_compression_ratio()
                        }
                    self.advanced_controls_data['compression_stats'] = stats
                    return jsonify({'status': 'success', 'devices': stats})
                else:
                    return jsonify({'error': 'Compression system not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/compression/compress', methods=['POST'])
        def compress_device_data():
            """Compress data on device."""
            try:
                if COMPRESSION_AVAILABLE:
                    data = request.get_json()
                    device_name = data.get('device_name')
                    if device_name:
                        result = auto_compress_device_data(device_name)
                        return jsonify({'status': 'success', 'result': result})
                    else:
                        return jsonify({'error': 'Device name required'})
                else:
                    return jsonify({'error': 'Compression system not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/compression/suggestions')
        def get_compression_suggestions():
            """Get compression suggestions."""
            try:
                if COMPRESSION_AVAILABLE:
                    suggestions = get_device_compression_suggestions()
                    return jsonify({'status': 'success', 'suggestions': suggestions})
                else:
                    return jsonify({'error': 'Compression system not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/api-keys/status')
        def get_api_keys_status():
            """Get API keys status."""
            try:
                if API_KEYS_AVAILABLE:
                    status = get_api_key_status()
                    exchanges = get_configured_exchanges()
                    self.dashboard_data['api_keys_configured'] = len(exchanges) > 0
                    return jsonify({
                        'status': 'success',
                        'configured': len(exchanges) > 0,
                        'exchanges': exchanges
                    })
                else:
                    return jsonify({'error': 'API key management not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/visual-controls/status')
        def get_visual_controls_status():
            """Get visual controls status."""
            try:
                if VISUAL_CONTROLS_AVAILABLE and self.visual_controls:
                    # Get current visual settings
                    settings = {
                        'chart_config': self.visual_controls.chart_config.__dict__,
                        'active_layers': [layer.value for layer in self.visual_controls.active_layers],
                        'auto_refresh': self.visual_controls.auto_refresh,
                        'refresh_interval': self.visual_controls.refresh_interval
                    }
                    self.advanced_controls_data['visual_settings'] = settings
                    return jsonify({'status': 'success', 'settings': settings})
                else:
                    return jsonify({'error': 'Visual controls not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/visual-controls/update', methods=['POST'])
        def update_visual_controls():
            """Update visual controls settings."""
            try:
                if VISUAL_CONTROLS_AVAILABLE and self.visual_controls:
                    data = request.get_json()
                    
                    # Update chart configuration
                    if 'chart_config' in data:
                        for key, value in data['chart_config'].items():
                            if hasattr(self.visual_controls.chart_config, key):
                                setattr(self.visual_controls.chart_config, key, value)
                    
                    # Update auto refresh
                    if 'auto_refresh' in data:
                        self.visual_controls.auto_refresh = data['auto_refresh']
                    
                    # Update refresh interval
                    if 'refresh_interval' in data:
                        self.visual_controls.refresh_interval = data['refresh_interval']
                    
                    return jsonify({'status': 'success', 'message': 'Visual controls updated'})
                else:
                    return jsonify({'error': 'Visual controls not available'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/launch/advanced-options')
        def launch_advanced_options():
            """Launch advanced options GUI."""
            try:
                subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/advanced_options_gui.py"])
                return jsonify({'status': 'success', 'message': 'Advanced options GUI launched'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/launch/visual-controls')
        def launch_visual_controls():
            """Launch visual controls GUI."""
            try:
                subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/visual_controls_gui.py"])
                return jsonify({'status': 'success', 'message': 'Visual controls GUI launched'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/launch/launcher')
        def launch_main_launcher():
            """Launch main Schwabot launcher."""
            try:
                subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/schwabot_launcher.py"])
                return jsonify({'status': 'success', 'message': 'Main launcher launched'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        # === SYSTEM CONTROL ENDPOINTS ===
        
        @app.route('/api/system/start', methods=['POST'])
        def start_system():
            """Start the dynamic timing system."""
            try:
                if self.dynamic_timing:
                    success = self.dynamic_timing.start()
                    if success:
                        self.dashboard_data['system_status'] = 'running'
                        return jsonify({'status': 'success', 'message': 'System started'})
                    else:
                        return jsonify({'status': 'error', 'message': 'Failed to start system'})
                else:
                    # Demo mode
                    self.dashboard_data['system_status'] = 'demo'
                    return jsonify({'status': 'success', 'message': 'Demo mode started'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @app.route('/api/system/stop', methods=['POST'])
        def stop_system():
            """Stop the dynamic timing system."""
            try:
                if self.dynamic_timing:
                    self.dynamic_timing.stop()
                self.dashboard_data['system_status'] = 'stopped'
                return jsonify({'status': 'success', 'message': 'System stopped'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @app.route('/api/explain/decision')
        def explain_decision():
            """Explain the current trading decision."""
            try:
                explanation = self.generate_decision_explanation()
                return jsonify(explanation)
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/api/advanced/controls/status')
        def get_all_advanced_controls_status():
            """Get status of all advanced control systems."""
            try:
                status = {
                    'usb_available': USB_AVAILABLE,
                    'compression_available': COMPRESSION_AVAILABLE,
                    'api_keys_available': API_KEYS_AVAILABLE,
                    'visual_controls_available': VISUAL_CONTROLS_AVAILABLE,
                    'advanced_controls_data': self.advanced_controls_data
                }
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)})

        # Add these new endpoints after the existing advanced control endpoints

        @app.route('/api/performance/status')
        def get_performance_status():
            """Get 2025 performance optimization status."""
            try:
                if PERFORMANCE_OPTIMIZER_AVAILABLE:
                    metrics = performance_optimizer.get_performance_metrics()
                    return jsonify({
                        'status': 'success',
                        'optimization_active': performance_optimizer.optimization_active,
                        'current_profile': {
                            'level': performance_optimizer.current_profile.optimization_level.value if performance_optimizer.current_profile else 'unknown',
                            'acceleration': performance_optimizer.current_profile.acceleration_type.value if performance_optimizer.current_profile else 'unknown',
                            'max_trades': performance_optimizer.current_profile.max_concurrent_trades if performance_optimizer.current_profile else 0,
                            'max_charts': performance_optimizer.current_profile.max_charts_per_device if performance_optimizer.current_profile else 0,
                            'latency_ms': performance_optimizer.current_profile.data_processing_latency_ms if performance_optimizer.current_profile else 0
                        },
                        'metrics': metrics
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'Performance optimizer not available'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @app.route('/api/performance/optimize', methods=['POST'])
        def optimize_performance():
            """Apply performance optimizations."""
            try:
                if PERFORMANCE_OPTIMIZER_AVAILABLE:
                    # Get hardware info
                    from core.hardware_auto_detector import hardware_detector
                    hw_info = hardware_detector.detect_hardware()
                    hw_dict = {
                        'ram_gb': hw_info.ram_gb,
                        'cpu_cores': hw_info.cpu_cores,
                        'gpu_memory_gb': hw_info.gpu.memory_gb if hasattr(hw_info, 'gpu') else 0.0
                    }
                    
                    # Detect and apply optimal profile
                    optimal_profile = performance_optimizer.detect_optimal_profile(hw_dict)
                    optimizations = performance_optimizer.optimize_system(optimal_profile)
                    
                    return jsonify({
                        'status': 'success',
                        'profile': {
                            'level': optimal_profile.optimization_level.value,
                            'acceleration': optimal_profile.acceleration_type.value,
                            'max_trades': optimal_profile.max_concurrent_trades,
                            'max_charts': optimal_profile.max_charts_per_device,
                            'latency_ms': optimal_profile.data_processing_latency_ms
                        },
                        'optimizations': optimizations
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'Performance optimizer not available'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @app.route('/api/performance/metrics')
        def get_performance_metrics():
            """Get real-time performance metrics."""
            try:
                if PERFORMANCE_OPTIMIZER_AVAILABLE:
                    metrics = performance_optimizer.get_performance_metrics()
                    return jsonify({
                        'status': 'success',
                        'metrics': metrics
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'Performance optimizer not available'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @app.route('/api/system/health')
        def get_system_health():
            """Get complete system health status."""
            try:
                import psutil
                
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Determine health status
                if cpu_usage < 80 and memory.percent < 80 and disk.percent < 90:
                    health_status = 'healthy'
                elif cpu_usage < 90 and memory.percent < 90 and disk.percent < 95:
                    health_status = 'warning'
                else:
                    health_status = 'critical'
                
                return jsonify({
                    'status': 'success',
                    'health': health_status,
                    'metrics': {
                        'cpu_usage_percent': cpu_usage,
                        'memory_usage_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_usage_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3)
                    },
                    'performance_optimization': {
                        'active': PERFORMANCE_OPTIMIZER_AVAILABLE and performance_optimizer.optimization_active,
                        'level': performance_optimizer.current_profile.optimization_level.value if (PERFORMANCE_OPTIMIZER_AVAILABLE and performance_optimizer.current_profile) else 'unknown'
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @app.route('/api/system/recommendations')
        def get_system_recommendations():
            """Get system optimization recommendations."""
            try:
                import psutil
                
                recommendations = []
                
                # Check CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > 80:
                    recommendations.append({
                        'type': 'warning',
                        'category': 'cpu',
                        'message': f'High CPU usage ({cpu_usage:.1f}%). Consider reducing concurrent operations.',
                        'priority': 'high'
                    })
                
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    recommendations.append({
                        'type': 'warning',
                        'category': 'memory',
                        'message': f'High memory usage ({memory.percent:.1f}%). Consider increasing RAM or optimizing memory usage.',
                        'priority': 'high'
                    })
                
                # Check disk usage
                disk = psutil.disk_usage('/')
                if disk.percent > 90:
                    recommendations.append({
                        'type': 'critical',
                        'category': 'storage',
                        'message': f'Low disk space ({disk.free / (1024**3):.1f}GB free). Consider cleanup or expansion.',
                        'priority': 'critical'
                    })
                
                # Performance optimization recommendations
                if PERFORMANCE_OPTIMIZER_AVAILABLE and performance_optimizer.current_profile:
                    profile = performance_optimizer.current_profile
                    
                    if profile.optimization_level.value == 'minimal':
                        recommendations.append({
                            'type': 'info',
                            'category': 'performance',
                            'message': 'System running on minimal optimization. Consider upgrading hardware for better performance.',
                            'priority': 'medium'
                        })
                    
                    if profile.acceleration_type.value == 'cpu_only':
                        recommendations.append({
                            'type': 'info',
                            'category': 'gpu',
                            'message': 'GPU acceleration not available. Consider adding a compatible GPU for better performance.',
                            'priority': 'medium'
                        })
                
                return jsonify({
                    'status': 'success',
                    'recommendations': recommendations,
                    'total_recommendations': len(recommendations)
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
    
    def start_data_collection(self):
        """Start data collection thread."""
        def data_collection_loop():
            while True:
                try:
                    self.update_dashboard_data()
                    self.broadcast_updates()
                    time.sleep(1.0)  # Update every second
                except Exception as e:
                    print(f"Error in data collection: {e}")
                    time.sleep(5.0)
        
        thread = threading.Thread(target=data_collection_loop, daemon=True)
        thread.start()
    
    def update_dashboard_data(self):
        """Update dashboard data from dynamic timing system."""
        try:
            if self.dynamic_timing and self.dynamic_timing.active:
                # Get real system status
                status = self.dynamic_timing.get_system_status()
                
                self.dashboard_data.update({
                    'rolling_profit': status.get('rolling_profit', 0.0),
                    'timing_accuracy': status.get('timing_accuracy', 0.0),
                    'total_signals': status.get('total_signals', 0),
                    'successful_signals': status.get('successful_signals', 0),
                    'system_uptime': status.get('uptime', 0),
                    'current_volatility': status.get('current_volatility', 0.0),
                    'current_momentum': status.get('current_momentum', 0.0),
                    'current_regime': status.get('current_regime', 'normal'),
                    'last_update': datetime.now().isoformat()
                })
                
                # Add to historical data
                current_time = time.time()
                self.historical_data['profit'].append({
                    'timestamp': current_time,
                    'value': self.dashboard_data['rolling_profit']
                })
                self.historical_data['volatility'].append({
                    'timestamp': current_time,
                    'value': self.dashboard_data['current_volatility']
                })
                self.historical_data['momentum'].append({
                    'timestamp': current_time,
                    'value': self.dashboard_data['current_momentum']
                })
                
                # Limit historical data
                max_points = 100
                for key in ['profit', 'volatility', 'momentum']:
                    if len(self.historical_data[key]) > max_points:
                        self.historical_data[key] = self.historical_data[key][-max_points:]
                
                # Generate recommendations
                self.update_recommendations()
                
            else:
                # Demo mode - generate simulated data
                self.generate_demo_data()
                
        except Exception as e:
            print(f"Error updating dashboard data: {e}")
    
    def generate_demo_data(self):
        """Generate demo data for visualization."""
        try:
            current_time = time.time()
            
            # Simulate profit with some randomness
            if not self.historical_data['profit']:
                base_profit = 0.0
            else:
                base_profit = self.historical_data['profit'][-1]['value']
            
            profit_change = random.uniform(-0.001, 0.002)
            new_profit = base_profit + profit_change
            
            # Simulate volatility and momentum
            volatility = random.uniform(0.001, 0.05)
            momentum = random.uniform(-0.02, 0.02)
            
            # Update dashboard data
            self.dashboard_data.update({
                'rolling_profit': new_profit,
                'timing_accuracy': random.uniform(0.6, 0.9),
                'total_signals': self.dashboard_data['total_signals'] + random.randint(0, 2),
                'successful_signals': self.dashboard_data['successful_signals'] + random.randint(0, 1),
                'system_uptime': self.dashboard_data['system_uptime'] + 1,
                'current_volatility': volatility,
                'current_momentum': momentum,
                'current_regime': self.determine_demo_regime(volatility, momentum),
                'last_update': datetime.now().isoformat()
            })
            
            # Add to historical data
            self.historical_data['profit'].append({
                'timestamp': current_time,
                'value': new_profit
            })
            self.historical_data['volatility'].append({
                'timestamp': current_time,
                'value': volatility
            })
            self.historical_data['momentum'].append({
                'timestamp': current_time,
                'value': momentum
            })
            
            # Limit historical data
            max_points = 100
            for key in ['profit', 'volatility', 'momentum']:
                if len(self.historical_data[key]) > max_points:
                    self.historical_data[key] = self.historical_data[key][-max_points:]
            
            # Generate recommendations
            self.update_recommendations()
            
        except Exception as e:
            print(f"Error generating demo data: {e}")
    
    def determine_demo_regime(self, volatility: float, momentum: float) -> str:
        """Determine demo regime based on volatility and momentum."""
        try:
            if volatility > 0.1 or abs(momentum) > 0.05:
                return 'crisis'
            elif volatility > 0.05:
                return 'extreme'
            elif volatility > 0.02:
                return 'volatile'
            elif volatility < 0.005 and abs(momentum) < 0.005:
                return 'calm'
            else:
                return 'normal'
        except Exception as e:
            print(f"Error determining demo regime: {e}")
            return 'normal'
    
    def update_recommendations(self):
        """Update trading recommendations."""
        try:
            profit = self.dashboard_data['rolling_profit']
            volatility = self.dashboard_data['current_volatility']
            momentum = self.dashboard_data['current_momentum']
            regime = self.dashboard_data['current_regime']
            
            # Generate recommendation based on current conditions
            recommendation = self.generate_recommendation(profit, volatility, momentum, regime)
            
            # Add timestamp
            recommendation['timestamp'] = datetime.now().isoformat()
            
            # Add to recommendations list
            self.recommendations.append(recommendation)
            
            # Keep only recent recommendations
            if len(self.recommendations) > 20:
                self.recommendations = self.recommendations[-20:]
                
        except Exception as e:
            print(f"Error updating recommendations: {e}")
    
    def generate_recommendation(self, profit: float, volatility: float, momentum: float, regime: str) -> Dict[str, Any]:
        """Generate trading recommendation based on current conditions."""
        try:
            recommendation = {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': '',
                'risk_level': 'MEDIUM',
                'expected_outcome': 'Stable performance',
                'timeframe': 'Short-term'
            }
            
            # Analyze conditions and generate recommendation
            if regime == 'crisis':
                recommendation.update({
                    'action': 'EMERGENCY_STOP',
                    'confidence': 0.9,
                    'reasoning': 'Crisis regime detected - extreme market conditions',
                    'risk_level': 'EXTREME',
                    'expected_outcome': 'Protect capital',
                    'timeframe': 'Immediate'
                })
            elif regime == 'extreme':
                if momentum > 0.02:
                    recommendation.update({
                        'action': 'AGGRESSIVE_BUY',
                        'confidence': 0.7,
                        'reasoning': 'High volatility with positive momentum',
                        'risk_level': 'HIGH',
                        'expected_outcome': '2-5% profit potential',
                        'timeframe': 'Short-term'
                    })
                else:
                    recommendation.update({
                        'action': 'CONSERVATIVE_SELL',
                        'confidence': 0.6,
                        'reasoning': 'High volatility with negative momentum',
                        'risk_level': 'HIGH',
                        'expected_outcome': 'Capital preservation',
                        'timeframe': 'Short-term'
                    })
            elif regime == 'volatile':
                if profit > 0.01:
                    recommendation.update({
                        'action': 'TAKE_PROFIT',
                        'confidence': 0.8,
                        'reasoning': 'Good profit in volatile conditions',
                        'risk_level': 'MEDIUM',
                        'expected_outcome': 'Secure gains',
                        'timeframe': 'Immediate'
                    })
                else:
                    recommendation.update({
                        'action': 'WAIT',
                        'confidence': 0.6,
                        'reasoning': 'Volatile conditions, wait for better opportunity',
                        'risk_level': 'MEDIUM',
                        'expected_outcome': 'Avoid losses',
                        'timeframe': 'Short-term'
                    })
            elif regime == 'normal':
                if momentum > 0.01 and profit < 0.005:
                    recommendation.update({
                        'action': 'BUY',
                        'confidence': 0.7,
                        'reasoning': 'Positive momentum with room for growth',
                        'risk_level': 'LOW',
                        'expected_outcome': '1-3% profit potential',
                        'timeframe': 'Medium-term'
                    })
                elif momentum < -0.01 and profit > 0.005:
                    recommendation.update({
                        'action': 'SELL',
                        'confidence': 0.7,
                        'reasoning': 'Negative momentum, secure profits',
                        'risk_level': 'LOW',
                        'expected_outcome': 'Protect gains',
                        'timeframe': 'Short-term'
                    })
                else:
                    recommendation.update({
                        'action': 'HOLD',
                        'confidence': 0.8,
                        'reasoning': 'Stable conditions, maintain position',
                        'risk_level': 'LOW',
                        'expected_outcome': 'Steady performance',
                        'timeframe': 'Medium-term'
                    })
            elif regime == 'calm':
                recommendation.update({
                    'action': 'HOLD',
                    'confidence': 0.9,
                    'reasoning': 'Calm market conditions, low volatility',
                    'risk_level': 'LOW',
                    'expected_outcome': 'Stable performance',
                    'timeframe': 'Long-term'
                })
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Unable to generate recommendation',
                'risk_level': 'UNKNOWN',
                'expected_outcome': 'Unknown',
                'timeframe': 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_decision_explanation(self) -> Dict[str, Any]:
        """Generate human-friendly explanation of current decision."""
        try:
            current_rec = self.recommendations[-1] if self.recommendations else None
            if not current_rec:
                return {
                    'explanation': 'No current recommendation available',
                    'confidence': 0.0,
                    'factors': []
                }
            
            # Generate explanation
            explanation = f"The system recommends {current_rec['action'].replace('_', ' ').lower()} "
            explanation += f"with {current_rec['confidence']*100:.0f}% confidence. "
            explanation += f"{current_rec['reasoning']} "
            explanation += f"Expected outcome: {current_rec['expected_outcome']} "
            explanation += f"over {current_rec['timeframe']} timeframe."
            
            # Identify key factors
            factors = []
            if self.dashboard_data['current_volatility'] > 0.05:
                factors.append('High market volatility')
            if abs(self.dashboard_data['current_momentum']) > 0.02:
                factors.append('Strong market momentum')
            if self.dashboard_data['rolling_profit'] > 0.01:
                factors.append('Positive profit trend')
            if self.dashboard_data['timing_accuracy'] > 0.8:
                factors.append('High timing accuracy')
            
            return {
                'explanation': explanation,
                'confidence': current_rec['confidence'],
                'factors': factors,
                'recommendation': current_rec
            }
            
        except Exception as e:
            print(f"Error generating decision explanation: {e}")
            return {
                'explanation': 'Unable to generate explanation',
                'confidence': 0.0,
                'factors': []
            }
    
    def broadcast_updates(self):
        """Broadcast updates to connected clients."""
        try:
            socketio.emit('dashboard_update', self.dashboard_data)
            socketio.emit('historical_update', self.historical_data)
            if self.recommendations:
                socketio.emit('recommendation_update', self.recommendations[-1])
        except Exception as e:
            print(f"Error broadcasting updates: {e}")

# Initialize dashboard
dashboard = DynamicTimingDashboard()

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request."""
    emit('status_response', dashboard.dashboard_data)

@socketio.on('request_historical')
def handle_historical_request():
    """Handle historical data request."""
    emit('historical_response', dashboard.historical_data)

@socketio.on('request_recommendations')
def handle_recommendations_request():
    """Handle recommendations request."""
    emit('recommendations_response', dashboard.recommendations)

def run_dashboard(host='0.0.0.0', port=8080, debug=False):
    """Run the dashboard."""
    try:
        print(f"🌐 Starting Dynamic Timing Dashboard on http://{host}:{port}")
        socketio.run(app, host=host, port=port, debug=debug)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    run_dashboard() 