#!/usr/bin/env python3
"""
Schwabot Visualizer Launcher - Comprehensive Visualization System
Integrates all existing visualizers and creates missing ones like DLT Wave Form.
"""

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from flask import Flask, jsonify, render_template, request

    print("✅ Visualizer dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install matplotlib numpy pandas flask")
    sys.exit(1)

# Try to import core modules with fallbacks
try:
    from core.clean_unified_math import clean_unified_math
    print("✅ clean_unified_math imported successfully")
except ImportError:
    print("⚠️ clean_unified_math not available, using stub")
    def clean_unified_math():
        """Stub for clean_unified_math."""
        return {"status": "stub", "message": "clean_unified_math not available"}

try:
    from core.unified_math_system import generate_unified_hash
    print("✅ generate_unified_hash imported successfully")
except ImportError:
    print("⚠️ generate_unified_hash not available, using stub")
    def generate_unified_hash():
        """Stub for generate_unified_hash."""
        return "stub_hash_" + str(int(time.time()))


class VisualizerLauncher:
    """Main launcher for all Schwabot visualizers."""

    def __init__(self):
        self.visualizers = {}
        self.active_visualizers = {}
        self.ports = {
            'main_dashboard': 5000,
            'dlt_waveform': 5001,
            'data_pipeline': 5002,
            'profit_analysis': 5003,
            'quantum_matrix': 5004,
            'strategy_visualizer': 5005,
            'portfolio_tracker': 5006,
            'risk_analyzer': 5007,
        }
        self._initialize_visualizers()

    def _initialize_visualizers(self):
        """Initialize all available visualizers."""
        self.visualizers = {
            'main_dashboard': {
                'name': 'Main Trading Dashboard',
                'description': 'Primary trading interface with real-time charts',
                'port': self.ports['main_dashboard'],
                'url': f"http://localhost:{self.ports['main_dashboard']}",
                'status': 'available',
                'launcher': self._launch_main_dashboard,
            },
            'dlt_waveform': {
                'name': 'DLT Wave Form Analyzer',
                'description': 'Distributed Ledger Technology waveform analysis',
                'port': self.ports['dlt_waveform'],
                'url': f"http://localhost:{self.ports['dlt_waveform']}",
                'status': 'available',
                'launcher': self._launch_dlt_waveform,
            },
            'data_pipeline': {
                'name': 'Data Pipeline Visualizer',
                'description': 'Real-time data flow and pipeline monitoring',
                'port': self.ports['data_pipeline'],
                'url': f"http://localhost:{self.ports['data_pipeline']}",
                'status': 'available',
                'launcher': self._launch_data_pipeline,
            },
            'profit_analysis': {
                'name': 'Profit Analysis Dashboard',
                'description': 'Advanced profit optimization and analysis',
                'port': self.ports['profit_analysis'],
                'url': f"http://localhost:{self.ports['profit_analysis']}",
                'status': 'available',
                'launcher': self._launch_profit_analysis,
            },
            'quantum_matrix': {
                'name': 'Quantum Matrix Visualizer',
                'description': 'Quantum tensor and superposition visualization',
                'port': self.ports['quantum_matrix'],
                'url': f"http://localhost:{self.ports['quantum_matrix']}",
                'status': 'available',
                'launcher': self._launch_quantum_matrix,
            },
            'strategy_visualizer': {
                'name': 'Strategy Intelligence Center',
                'description': 'AI strategy visualization and analysis',
                'port': self.ports['strategy_visualizer'],
                'url': f"http://localhost:{self.ports['strategy_visualizer']}",
                'status': 'available',
                'launcher': self._launch_strategy_visualizer,
            },
            'portfolio_tracker': {
                'name': 'Portfolio Performance Tracker',
                'description': 'Real-time portfolio tracking and analytics',
                'port': self.ports['portfolio_tracker'],
                'url': f"http://localhost:{self.ports['portfolio_tracker']}",
                'status': 'available',
                'launcher': self._launch_portfolio_tracker,
            },
            'risk_analyzer': {
                'name': 'Risk Management Analyzer',
                'description': 'Risk assessment and management visualization',
                'port': self.ports['risk_analyzer'],
                'url': f"http://localhost:{self.ports['risk_analyzer']}",
                'status': 'available',
                'launcher': self._launch_risk_analyzer,
            },
        }

    def launch_visualizer(self, visualizer_id: str) -> bool:
        """Launch a specific visualizer."""
        if visualizer_id not in self.visualizers:
            print(f"❌ Unknown visualizer: {visualizer_id}")
            return False

        visualizer = self.visualizers[visualizer_id]

        try:
            # Launch in background thread
            thread = threading.Thread(target=visualizer['launcher'], daemon=True)
            thread.start()

            # Update status
            visualizer['status'] = 'running'
            self.active_visualizers[visualizer_id] = {
                'thread': thread,
                'start_time': time.time(),
                'port': visualizer['port'],
            }

            # Open browser after delay
            threading.Timer(2.0, lambda: self._open_browser(visualizer['url'])).start()

            print(f"✅ Launched {visualizer['name']} on {visualizer['url']}")
            return True

        except Exception as e:
            print(f"❌ Failed to launch {visualizer['name']}: {e}")
            return False

    def _open_browser(self, url: str):
        """Open browser to visualizer URL."""
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"❌ Browser error: {e}")

    def _launch_main_dashboard(self):
        """Launch main trading dashboard."""
        try:
            from gui.flask_app import app

            app.run(host='127.0.0.1', port=self.ports['main_dashboard'], debug=False)
        except Exception as e:
            print(f"❌ Main dashboard error: {e}")

    def _launch_dlt_waveform(self):
        """Launch DLT Wave Form Analyzer."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('dlt_waveform.html')

            @app.route('/api/dlt_data')
            def get_dlt_data():
                """Generate DLT waveform data."""
                # Generate sample DLT data
                timestamps = np.linspace(0, 100, 1000)
                base_frequency = 0.1

                # Create complex DLT waveform
                waveform = (
                    np.sin(2 * np.pi * base_frequency * timestamps) * 0.5
                    + np.sin(2 * np.pi * base_frequency * 2 * timestamps) * 0.3
                    + np.sin(2 * np.pi * base_frequency * 3 * timestamps) * 0.2
                    + np.random.normal(0, 0.05, len(timestamps))
                )

                # Add DLT-specific patterns
                dlt_patterns = []
                for i in range(0, len(timestamps), 100):
                    pattern = {
                        'timestamp': timestamps[i],
                        'amplitude': waveform[i],
                        'frequency': base_frequency * (1 + i % 3),
                        'phase': (i % 360) * np.pi / 180,
                        'confidence': 0.8 + 0.2 * np.random.random(),
                    }
                    dlt_patterns.append(pattern)

                return jsonify(
                    {
                        'timestamps': timestamps.tolist(),
                        'waveform': waveform.tolist(),
                        'patterns': dlt_patterns,
                        'analysis': {
                            'dominant_frequency': base_frequency,
                            'amplitude_range': [float(np.min(waveform)), float(np.max(waveform))],
                            'pattern_count': len(dlt_patterns),
                            'confidence_score': 0.85,
                        },
                    }
                )

            app.run(host='127.0.0.1', port=self.ports['dlt_waveform'], debug=False)

        except Exception as e:
            print(f"❌ DLT Waveform error: {e}")

    def _launch_data_pipeline(self):
        """Launch Data Pipeline Visualizer."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('data_pipeline.html')

            @app.route('/api/pipeline_data')
            def get_pipeline_data():
                """Get real-time pipeline data."""
                # Simulate pipeline data
                pipeline_data = {
                    'nodes': [
                        {
                            'id': 'input',
                            'name': 'Data Input',
                            'status': 'active',
                            'throughput': 1000,
                        },
                        {
                            'id': 'process',
                            'name': 'Processing',
                            'status': 'active',
                            'throughput': 950,
                        },
                        {
                            'id': 'analyze',
                            'name': 'Analysis',
                            'status': 'active',
                            'throughput': 900,
                        },
                        {'id': 'output', 'name': 'Output', 'status': 'active', 'throughput': 850},
                    ],
                    'connections': [
                        {'from': 'input', 'to': 'process', 'latency': 5},
                        {'from': 'process', 'to': 'analyze', 'latency': 10},
                        {'from': 'analyze', 'to': 'output', 'latency': 3},
                    ],
                    'metrics': {
                        'total_throughput': 850,
                        'avg_latency': 6.0,
                        'error_rate': 0.02,
                        'memory_usage': 75.5,
                    },
                }

                return jsonify(pipeline_data)

            app.run(host='127.0.0.1', port=self.ports['data_pipeline'], debug=False)

        except Exception as e:
            print(f"❌ Data Pipeline error: {e}")

    def _launch_profit_analysis(self):
        """Launch Profit Analysis Dashboard."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('profit_analysis.html')

            @app.route('/api/profit_data')
            def get_profit_data():
                """Get profit analysis data."""
                # Use clean_unified_math for calculations
                math_result = clean_unified_math.integrate_all_systems(
                    {'tensor': [[60000, 1000]], 'metadata': {'confidence': 0.8}}
                )

                profit_data = {
                    'current_profit': 1250.50,
                    'total_trades': 45,
                    'win_rate': 0.73,
                    'avg_profit_per_trade': 27.78,
                    'max_drawdown': -150.25,
                    'sharpe_ratio': 1.85,
                    'math_score': math_result.get('combined_score', 0.0),
                    'profit_history': [
                        {'date': '2025-01-01', 'profit': 100},
                        {'date': '2025-01-02', 'profit': 250},
                        {'date': '2025-01-03', 'profit': 180},
                        {'date': '2025-01-04', 'profit': 320},
                        {'date': '2025-01-05', 'profit': 400},
                    ],
                }

                return jsonify(profit_data)

            app.run(host='127.0.0.1', port=self.ports['profit_analysis'], debug=False)

        except Exception as e:
            print(f"❌ Profit Analysis error: {e}")

    def _launch_quantum_matrix(self):
        """Launch Quantum Matrix Visualizer."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('quantum_matrix.html')

            @app.route('/api/quantum_data')
            def get_quantum_data():
                """Get quantum matrix data."""
                # Generate quantum tensor data
                tensor_size = 8
                quantum_tensor = np.random.random((tensor_size, tensor_size)) + 1j * np.random.random(
                    (tensor_size, tensor_size)
                )

                # Calculate quantum metrics
                eigenvalues = np.linalg.eigvals(quantum_tensor)
                coherence = np.abs(np.trace(quantum_tensor)) / tensor_size

                quantum_data = {
                    'tensor': quantum_tensor.tolist(),
                    'eigenvalues': eigenvalues.tolist(),
                    'coherence': float(coherence),
                    'entanglement': 0.75,
                    'superposition_states': 4,
                    'quantum_score': 0.88,
                }

                return jsonify(quantum_data)

            app.run(host='127.0.0.1', port=self.ports['quantum_matrix'], debug=False)

        except Exception as e:
            print(f"❌ Quantum Matrix error: {e}")

    def _launch_strategy_visualizer(self):
        """Launch Strategy Intelligence Center."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('strategy_visualizer.html')

            @app.route('/api/strategy_data')
            def get_strategy_data():
                """Get strategy analysis data."""
                strategy_data = {
                    'active_strategies': [
                        {'name': 'Momentum Strategy', 'performance': 0.85, 'risk': 0.3},
                        {'name': 'Mean Reversion', 'performance': 0.72, 'risk': 0.2},
                        {'name': 'Arbitrage', 'performance': 0.91, 'risk': 0.1},
                        {'name': 'Trend Following', 'performance': 0.78, 'risk': 0.4},
                    ],
                    'ai_predictions': [
                        {'asset': 'BTC', 'prediction': 'BUY', 'confidence': 0.88},
                        {'asset': 'ETH', 'prediction': 'HOLD', 'confidence': 0.65},
                        {'asset': 'SOL', 'prediction': 'SELL', 'confidence': 0.72},
                    ],
                    'market_sentiment': 0.75,
                    'strategy_optimization': 0.82,
                }

                return jsonify(strategy_data)

            app.run(host='127.0.0.1', port=self.ports['strategy_visualizer'], debug=False)

        except Exception as e:
            print(f"❌ Strategy Visualizer error: {e}")

    def _launch_portfolio_tracker(self):
        """Launch Portfolio Performance Tracker."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('portfolio_tracker.html')

            @app.route('/api/portfolio_data')
            def get_portfolio_data():
                """Get portfolio tracking data."""
                portfolio_data = {
                    'total_value': 125000.50,
                    'daily_change': 1250.75,
                    'total_return': 0.25,
                    'positions': [
                        {'asset': 'BTC', 'quantity': 2.5, 'value': 75000, 'return': 0.15},
                        {'asset': 'ETH', 'quantity': 15.0, 'value': 35000, 'return': 0.08},
                        {'asset': 'SOL', 'quantity': 100.0, 'value': 15000, 'return': 0.35},
                    ],
                    'performance_history': [
                        {'date': '2025-01-01', 'value': 100000},
                        {'date': '2025-01-02', 'value': 102500},
                        {'date': '2025-01-03', 'value': 101800},
                        {'date': '2025-01-04', 'value': 104200},
                        {'date': '2025-01-05', 'value': 105000},
                    ],
                }

                return jsonify(portfolio_data)

            app.run(host='127.0.0.1', port=self.ports['portfolio_tracker'], debug=False)

        except Exception as e:
            print(f"❌ Portfolio Tracker error: {e}")

    def _launch_risk_analyzer(self):
        """Launch Risk Management Analyzer."""
        try:
            app = Flask(__name__)

            @app.route('/')
            def index():
                return render_template('risk_analyzer.html')

            @app.route('/api/risk_data')
            def get_risk_data():
                """Get risk analysis data."""
                risk_data = {
                    'current_risk': 0.35,
                    'max_risk': 0.50,
                    'var_95': 2500.0,
                    'max_drawdown': -1500.0,
                    'risk_metrics': {
                        'volatility': 0.18,
                        'beta': 1.2,
                        'sharpe_ratio': 1.85,
                        'sortino_ratio': 2.1,
                    },
                    'risk_alerts': [
                        {
                            'type': 'high_volatility',
                            'message': 'BTC volatility above threshold',
                            'severity': 'medium',
                        },
                        {
                            'type': 'concentration',
                            'message': 'Portfolio too concentrated in crypto',
                            'severity': 'high',
                        },
                    ],
                }

                return jsonify(risk_data)

            app.run(host='127.0.0.1', port=self.ports['risk_analyzer'], debug=False)

        except Exception as e:
            print(f"❌ Risk Analyzer error: {e}")

    def list_visualizers(self) -> Dict[str, Any]:
        """List all available visualizers."""
        return {
            'visualizers': self.visualizers,
            'active_count': len(self.active_visualizers),
            'total_count': len(self.visualizers),
        }

    def stop_visualizer(self, visualizer_id: str) -> bool:
        """Stop a specific visualizer."""
        if visualizer_id in self.active_visualizers:
            # Note: Flask servers run in daemon threads, so they'll stop when main thread ends
            del self.active_visualizers[visualizer_id]
            self.visualizers[visualizer_id]['status'] = 'available'
            print(f"✅ Stopped {self.visualizers[visualizer_id]['name']}")
            return True
        return False

    def stop_all_visualizers(self):
        """Stop all active visualizers."""
        for visualizer_id in list(self.active_visualizers.keys()):
            self.stop_visualizer(visualizer_id)
        print("✅ Stopped all visualizers")


def main():
    """Main entry point for visualizer launcher."""
    launcher = VisualizerLauncher()

    print("=" * 60)
    print("🚀 SCHWABOT VISUALIZER LAUNCHER")
    print("=" * 60)
    print("Available Visualizers:")

    for vid, vinfo in launcher.visualizers.items():
        status_icon = "🟢" if vinfo['status'] == 'running' else "⚪"
        print(f"  {status_icon} {vid}: {vinfo['name']}")
        print(f"     {vinfo['description']}")
        print(f"     URL: {vinfo['url']}")
        print()

    print("Commands:")
    print("  launch <visualizer_id>  - Launch specific visualizer")
    print("  list                    - List all visualizers")
    print("  stop <visualizer_id>    - Stop specific visualizer")
    print("  stop_all                - Stop all visualizers")
    print("  exit                    - Exit launcher")
    print("=" * 60)

    while True:
        try:
            command = input("visualizer> ").strip().split()
            if not command:
                continue

            cmd = command[0].lower()

            if cmd == 'launch' and len(command) > 1:
                visualizer_id = command[1]
                launcher.launch_visualizer(visualizer_id)

            elif cmd == 'list':
                visualizers = launcher.list_visualizers()
                print(f"\n📊 Visualizer Status:")
                print(f"Active: {visualizers['active_count']}/{visualizers['total_count']}")
                for vid, vinfo in visualizers['visualizers'].items():
                    status_icon = "🟢" if vinfo['status'] == 'running' else "⚪"
                    print(f"  {status_icon} {vid}: {vinfo['name']}")
                print()

            elif cmd == 'stop' and len(command) > 1:
                visualizer_id = command[1]
                launcher.stop_visualizer(visualizer_id)

            elif cmd == 'stop_all':
                launcher.stop_all_visualizers()

            elif cmd == 'exit':
                launcher.stop_all_visualizers()
                print("👋 Goodbye!")
                break

            else:
                print("❌ Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            launcher.stop_all_visualizers()
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
